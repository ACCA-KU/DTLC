"""Segment-contribution visualization helpers for BLI/CLI_GAT_SEI_visualization.ipynb."""
import html
import os
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import HTML, display


def _as_1d_score(score_value):
    return np.asarray(score_value, dtype=float).reshape(-1)


def _fragment_atom_indices(fragment):
    return [int(a) for a in fragment.atoms]


def _subgroup_scores_to_atom_scores(mol, fragments, subgroup_scores):
    atom_values = {i: [] for i in range(mol.GetNumAtoms())}

    for fragment, score_value in zip(fragments, subgroup_scores):
        for atom_idx in _fragment_atom_indices(fragment):
            atom_values[atom_idx].append(float(score_value))

    return {
        atom_idx: float(np.mean(values))
        for atom_idx, values in atom_values.items()
        if values
    }


def _blue_white_red(score_value, vmin, vmax):
    max_val = max(abs(vmin), abs(vmax))
    
    if np.isclose(max_val, 0.0):
        return (0.95, 0.95, 0.95)

    new_vmin= -max_val
    new_vmax = max_val

    t = float((score_value - new_vmin) / (new_vmax - new_vmin))
    t = min(max(t, 0.0), 1.0)

    if t < 0.5:
        # blue -> white
        u = t / 0.5
        return (u, u, 1.0)

    # white -> red
    u = (t - 0.5) / 0.5
    return (1.0, 1.0 - u, 1.0 - u)


def _draw_attention_svg(smiles, fragments, subgroup_scores, title, vmin, vmax, width=360, height=280):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid molecule SMILES: {smiles}")

    mol = Chem.Mol(mol)
    atom_scores = _subgroup_scores_to_atom_scores(mol, fragments, subgroup_scores)
    highlight_atoms = list(atom_scores.keys())
    highlight_colors = {
        atom_idx: _blue_white_red(atom_scores[atom_idx], vmin, vmax)
        for atom_idx in highlight_atoms
    }
    highlight_radii = {
        atom_idx: 0.32 + 0.28 * abs((atom_scores[atom_idx] - vmin) / (vmax - vmin))
        if not np.isclose(vmin, vmax) else 0.35
        for atom_idx in highlight_atoms
    }

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.addAtomIndices = True
    drawer.DrawMolecule(
        mol,
        legend=title,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightAtomRadii=highlight_radii,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace("svg:", "")


def _score_table_html(fragments, subgroup_scores, score_key, eluent_smiles=None):
    rows = []
    if eluent_smiles is not None:
        rows.append(
            f"<div style='margin: 6px 0 8px 0;'><b>eluent smiles</b>: "
            f"<code>{html.escape(str(eluent_smiles))}</code></div>"
        )

    rows.append("<table style='border-collapse: collapse; font-size: 12px;'>")
    rows.append(
        "<tr>"
        "<th style='text-align:right; padding:2px 8px;'>frag</th>"
        "<th style='text-align:left; padding:2px 8px;'>frag_smiles</th>"
        "<th style='text-align:left; padding:2px 8px;'>atoms</th>"
        f"<th style='text-align:right; padding:2px 8px;'>{html.escape(score_key)}</th>"
        "</tr>"
    )
    for idx, (fragment, score_value) in enumerate(zip(fragments, subgroup_scores)):
        rows.append(
            "<tr>"
            f"<td style='text-align:right; padding:2px 8px;'>{idx}</td>"
            f"<td style='padding:2px 8px;'><code>{html.escape(str(fragment.smiles))}</code></td>"
            f"<td style='padding:2px 8px;'>{_fragment_atom_indices(fragment)}</td>"
            f"<td style='text-align:right; padding:2px 8px;'>{float(score_value):.6g}</td>"
            "</tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _attention_keys(score, ratio=False):
    if "attn1" in score and "attn2" in score:
        keys = ("attn1_ratio", "attn2_ratio") if ratio else ("attn1", "attn2")
        missing = [key for key in keys if key not in score]
        if missing:
            raise KeyError(f"Missing score key(s): {missing}. Available keys: {list(score.keys())}")
        return list(keys)

    if ratio and "attn_ratio" in score:
        return ["attn_ratio"]
    if "attn" in score:
        # Some BLI segment models already fold the eluent ratio into a single attn value.
        return ["attn"]

    expected = "attn_ratio" if ratio else "attn"
    raise KeyError(f"Missing score key: {expected}. Available keys: {list(score.keys())}")


def _contribution_keys(score, ratio=False):
    if "subgroup_contribution1" in score and "subgroup_contribution2" in score:
        keys = (
            ("ratio_weighted_contribution1", "ratio_weighted_contribution2")
            if ratio else
            ("subgroup_contribution1", "subgroup_contribution2")
        )
        missing = [key for key in keys if key not in score]
        if missing:
            raise KeyError(f"Missing score key(s): {missing}. Available keys: {list(score.keys())}")
        return list(keys)

    if "subgroup_contributions" in score:
        # BLI sum-contribution output already uses the ratio-weighted bulk eluent representation.
        return ["subgroup_contributions"]

    raise KeyError(
        "Missing sum-contribution score keys. "
        f"Available keys: {list(score.keys())}"
    )


def visualize_segment_attention2(
    analyzer,
    smiles,
    eluent1_smiles,
    eluent2_smiles,
    ratio1,
    ratio2,
    *,
    ratio=False,
    score=None,
    width=360,
    height=280,
):
    """Visualize subgroup contribution returned by sum-contribution models.

    ratio=False uses raw subgroup contribution keys.
    ratio=True uses ratio-weighted contribution keys when the model returns separate eluent outputs.
    """
    if score is None:
        score = analyzer.get_score(smiles, eluent1_smiles, eluent2_smiles, ratio1, ratio2)

    fragments = analyzer.get_fragment(smiles)
    keys = _contribution_keys(score, ratio=ratio)
    score_arrays = [_as_1d_score(score[key]) for key in keys]

    expected_size = len(fragments)
    for key, values in zip(keys, score_arrays):
        if len(values) != expected_size:
            raise ValueError(
                f"{key} length ({len(values)}) does not match fragment count ({expected_size})."
            )

    all_values = np.concatenate(score_arrays)
    vmin = float(np.min(all_values))
    vmax = float(np.max(all_values))

    combined_eluent_label = (
        f"{eluent1_smiles} ({ratio1:g}) + {eluent2_smiles} ({ratio2:g})"
    )
    eluent_by_key = {
        "subgroup_contributions": combined_eluent_label,
        "subgroup_contribution1": eluent1_smiles,
        "ratio_weighted_contribution1": eluent1_smiles,
        "subgroup_contribution2": eluent2_smiles,
        "ratio_weighted_contribution2": eluent2_smiles,
    }

    panels = []
    for key, values in zip(keys, score_arrays):
        eluent_smiles = eluent_by_key.get(key)
        title = key if eluent_smiles is None else f"{key}: {eluent_smiles}"
        svg = _draw_attention_svg(
            smiles,
            fragments,
            values,
            title=title,
            vmin=vmin,
            vmax=vmax,
            width=width,
            height=height,
        )
        table = _score_table_html(fragments, values, key, eluent_smiles=eluent_smiles)
        panels.append(
            "<div style='display:inline-block; vertical-align:top; margin:0 18px 18px 0;'>"
            f"{svg}"
            f"{table}"
            "</div>"
        )

    caption = (
        "<div style='margin-bottom:8px; font-size:13px;'>"
        "Color scale: low contribution = blue, high contribution = red. "
        f"Range: {vmin:.6g} to {vmax:.6g}."
        "</div>"
    )
    display(HTML(caption + "<div>" + "\n".join(panels) + "</div>"))
    return score


# ===== ratio sweep (BLI-GAT-SEI) =====

def _get_score_quiet(analyzer, *args, verbose=False):
    if verbose:
        return analyzer.get_score(*args)
    with open(os.devnull, "w") as null, redirect_stdout(null):
        return analyzer.get_score(*args)


def visualize_ratio_sweep(
    analyzer,
    smiles,
    *,
    hex_smiles="CCCCCC",
    ea_smiles="CCOC(C)=O",
    ea_fractions=None,
    label1="HEX",
    label2="EA",
    ratio=False,
    shared_scale=True,
    show_tables=False,
    width=300,
    height=240,
    verbose=False,
):
    
    if ea_fractions is None:
        ea_fractions = np.round(np.linspace(0.0, 1.0, 11), 1)  # 0.0, 0.1, ..., 1.0

    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    fragments = analyzer.get_fragment(smiles)

    scores = {}
    for ea in ea_fractions:
        hex_ = round(1.0 - ea, 10)
        
        scores[ea] = _get_score_quiet(analyzer, smiles, hex_smiles, ea_smiles, hex_, ea, verbose=verbose)

    # normalization
    contribution_keys = _contribution_keys(next(iter(scores.values())), ratio=ratio)
    raw_scores = {ea: dict(score) for ea, score in scores.items()}
    for score in scores.values():
        for key in contribution_keys:
            score[key] = np.tanh(_as_1d_score(score[key]))

    if not shared_scale:
        for ea, score in scores.items():
            visualize_segment_attention2(
                analyzer, smiles, hex_smiles, ea_smiles, round(1.0 - ea, 10), ea,
                ratio=ratio, score=score, width=width, height=height,
            )
    else:
        keys = _contribution_keys(next(iter(scores.values())), ratio=ratio)
        arrays = {
            ea: [_as_1d_score(score[key]) for key in keys] for ea, score in scores.items()
        }
        all_values = np.concatenate([v for vals in arrays.values() for v in vals])
        vmin, vmax = float(all_values.min()), float(all_values.max())

        panels = []
        for ea, score in scores.items():
            rf = float(analyzer.scaler.inverse_transform(
                np.asarray(score["y"], dtype=float).reshape(-1, 1))[0, 0])
            for key, values in zip(keys, arrays[ea]):
                suffix = "" if len(keys) == 1 else f" [{key}]"
                title = f"{label2}/{label1} = {ea:.1f}/{1 - ea:.1f}  |  Rf = {rf:.5f}{suffix}"
                svg = _draw_attention_svg(
                    smiles, fragments, values, title=title,
                    vmin=vmin, vmax=vmax, width=width, height=height,
                )
                table = _score_table_html(fragments, values, key) if show_tables else ""
                panels.append(
                    "<div style='display:inline-block; vertical-align:top; margin:0 12px 12px 0;'>"
                    f"{svg}{table}</div>"
                )

        caption = (
            "<div style='margin-bottom:8px; font-size:13px;'>"
            f"<b>{html.escape(smiles)}</b> — "
            "Color scale : low contribution(to -1) = blue, high contribution(to +1) = red. "
            f"Range: {vmin:.6g} to {vmax:.6g}."
            "</div>"
        )
        display(HTML(caption + "<div>" + "\n".join(panels) + "</div>"))

    # tables : fragment * contribution
    keys = _contribution_keys(next(iter(scores.values())), ratio=ratio)
    rows = []
    for ea, score in scores.items():
        rf = float(analyzer.scaler.inverse_transform(
            np.asarray(score["y"], dtype=float).reshape(-1, 1))[0, 0])
        for key in keys:
            raw_values = _as_1d_score(raw_scores[ea][key])
            for idx, (fragment, value) in enumerate(zip(fragments, _as_1d_score(score[key]))):
                rows.append({
                    label2: ea, label1: round(1.0 - ea, 10), "Rf_pred": rf, "key": key,
                    "frag": idx, "frag_smiles": fragment.smiles,
                    "contribution": float(value),  # tanh
                    "contribution_raw": float(raw_values[idx]),
                })
    table = pd.DataFrame(rows)
    return table, scores


# ===== ratio sweep (CLI-GAT-SEI) =====

CLI_MODE_KEYS = {
    "ratio_weighted": ["ratio_weighted_contribution1", "ratio_weighted_contribution2"]
}


def visualize_ratio_sweep_cli(
    analyzer,
    smiles,
    *,
    hex_smiles="CCCCCC",
    ea_smiles="CCOC(C)=O",
    ea_fractions=None,
    label1="HEX",
    label2="EA",
    mode="total",
    shared_scale=True,
    show_tables=False,
    width=300,
    height=240,
    verbose=False,
):
    if ea_fractions is None:
        ea_fractions = np.round(np.linspace(0.0, 1.0, 11), 1)  # 0.0, 0.1, ..., 1.0
    keys = CLI_MODE_KEYS[mode]
    key_labels = {
        "total_contribution": "",
        "ratio_weighted_contribution1": f" [{label1}]",
        "ratio_weighted_contribution2": f" [{label2}]",
        "subgroup_contribution1": f" [raw {label1}]",
        "subgroup_contribution2": f" [raw {label2}]",
    }

    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    fragments = analyzer.get_fragment(smiles)

    scores = {}
    for ea in ea_fractions:
        hex_ = round(1.0 - ea, 10)
        score = _get_score_quiet(analyzer, smiles, hex_smiles, ea_smiles, hex_, ea, verbose=verbose)
        score["total_contribution"] = (
            _as_1d_score(score["ratio_weighted_contribution1"])
            + _as_1d_score(score["ratio_weighted_contribution2"])
        )
        scores[ea] = score

    # normalization
    raw_scores = {ea: dict(score) for ea, score in scores.items()}
    for score in scores.values():
        for key in keys:
            score[key] = np.tanh(_as_1d_score(score[key]))

    arrays = {ea: [_as_1d_score(score[key]) for key in keys] for ea, score in scores.items()}
    if shared_scale:
        all_values = np.concatenate([v for vals in arrays.values() for v in vals])
        vmin, vmax = float(all_values.min()), float(all_values.max())

    rows_html = []
    for ea, score in scores.items():
        rf = float(analyzer.scaler.inverse_transform(
            np.asarray(score["y"], dtype=float).reshape(-1, 1))[0, 0])
        if not shared_scale:
            ratio_values = np.concatenate(arrays[ea])
            vmin, vmax = float(ratio_values.min()), float(ratio_values.max())
        panels = []
        for key, values in zip(keys, arrays[ea]):
            title = f"{label2}/{label1} = {ea:.1f}/{1 - ea:.1f}  |  Rf = {rf:.5f}{key_labels[key]}"
            svg = _draw_attention_svg(
                smiles, fragments, values, title=title,
                vmin=vmin, vmax=vmax, width=width, height=height,
            )
            table = _score_table_html(fragments, values, key) if show_tables else ""
            panels.append(
                "<div style='display:inline-block; vertical-align:top; margin:0 12px 12px 0;'>"
                f"{svg}{table}</div>"
            )
        rows_html.append("\n".join(panels))

    joiner = "\n" if len(keys) == 1 else "<br>\n"
    caption = (
        "<div style='margin-bottom:8px; font-size:13px;'>"
        f"<b>{html.escape(smiles)}</b> — mode = {mode}, color scale: "
        "low contribution(to -1) = blue, high contribution(to +1) = red. "
        + (f"Range: {vmin:.6g} to {vmax:.6g}." if shared_scale else "")
        + "</div>"
    )
    display(HTML(caption + "<div>" + joiner.join(rows_html) + "</div>"))

    # tables : fragment * contribution
    rows = []
    for ea, score in scores.items():
        rf = float(analyzer.scaler.inverse_transform(
            np.asarray(score["y"], dtype=float).reshape(-1, 1))[0, 0])
        for key in keys:
            raw_values = _as_1d_score(raw_scores[ea][key])
            for idx, (fragment, value) in enumerate(zip(fragments, _as_1d_score(score[key]))):
                rows.append({
                    label2: ea, label1: round(1.0 - ea, 10), "Rf_pred": rf, "key": key,
                    "frag": idx, "frag_smiles": fragment.smiles,
                    "contribution": float(value),  # tanh
                    "contribution_raw": float(raw_values[idx]),
                })
    table = pd.DataFrame(rows)
    return table, scores

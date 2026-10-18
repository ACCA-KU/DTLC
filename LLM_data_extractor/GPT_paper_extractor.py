"""Utilities for extracting TLC Rf data from supporting-information PDFs.

The public functions in this module are intended to be called from
``GPT_paper_extract.ipynb``. OpenAI credentials are read from the
``OPENAI_API_KEY`` environment variable by the OpenAI client.
"""

from __future__ import annotations

import base64
import difflib
import io
import json
import re
import subprocess
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import openai
import pandas as pd
from IPython.display import display
from openai import OpenAI
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text
from rich import print as richprint
from rich.console import Console
from tqdm.auto import tqdm


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_SI_DIR = MODULE_DIR / "SI"
DEFAULT_OUTPUT_DIR = MODULE_DIR / "data_extracted_gpt"
DEFAULT_OPSIN_JAR = MODULE_DIR / "opsin-cli-2.8.0-jar-with-dependencies.jar"
DEFAULT_MODEL = "gpt-4.1-mini"

EXTRACTED_COLUMNS = [
    "compound name",
    "compound IUPAC name",
    "eluent1",
    "eluent2",
    "ratio",
    "Rf",
]


def make_buffer_page(page: Sequence[bool]) -> list[bool]:
    """Include pages immediately adjacent to pages that mention an Rf value."""
    new_page = list(page)
    for index, is_rf_page in enumerate(page):
        if not is_rf_page:
            left = page[index - 1] if index > 0 else False
            right = page[index + 1] if index < len(page) - 1 else False
            if left or right:
                new_page[index] = True
    return new_page


def pdf_Rf_page_img(
    pdf_list: Sequence[str | Path],
    *,
    poppler_path: str | Path | None = None,
    render_images: bool = False,
) -> tuple[list[list[Any]], str, list[str], list[list[list[str]]]]:
    """Find Rf-related pages and optionally render their PDF pages as images."""
    images_by_pdf: list[list[Any]] = []
    all_rf_page_text: list[str] = []
    all_sentence_windows: list[list[list[str]]] = []

    for pdf_index, pdf in enumerate(pdf_list):
        pdf_path = Path(pdf)
        print(f"pdf {pdf_index}: {pdf_path.name}")
        text = extract_text(str(pdf_path))

        if "\x0c" not in text:
            print(f"{pdf_path}: no PDF page separator (\\x0c) was detected.")
            continue

        pages_text = text.split("\x0c")
        rf_pages_bool = [("Rf" in page or "RF" in page) for page in pages_text]
        rf_pages_num = np.flatnonzero(make_buffer_page(rf_pages_bool)).tolist()
        rf_pages_text = [pages_text[index] for index in rf_pages_num]
        all_rf_page_text.extend(rf_pages_text)

        print("Checking each page...")
        for page_text in tqdm(rf_pages_text):
            lines = [
                line
                for line in page_text.split("\n")
                if not re.fullmatch(r"\s*", line)
            ]
            nearby_indices = [
                [index - 1, index, index + 1, index + 2]
                for index, line in enumerate(lines)
                if "Rf" in line or "RF" in line
            ]
            all_sentence_windows.append(
                [
                    [lines[index] for index in indices if 0 <= index < len(lines)]
                    for indices in nearby_indices
                ]
            )

        if render_images:
            kwargs = {}
            if poppler_path is not None:
                kwargs["poppler_path"] = str(poppler_path)
            rendered_pages = convert_from_path(str(pdf_path), **kwargs)
            images_by_pdf.append(
                [rendered_pages[index] for index in rf_pages_num]
            )
        else:
            images_by_pdf.append([])

    return (
        images_by_pdf,
        "\x0c".join(all_rf_page_text),
        all_rf_page_text,
        all_sentence_windows,
    )


def get_img_uri(img: Any) -> str:
    """Convert a PIL image to a base64 PNG data URI."""
    png_buffer = io.BytesIO()
    img.save(png_buffer, format="PNG")
    png_buffer.seek(0)
    base64_png = base64.b64encode(png_buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{base64_png}"


def get_img_markdown(img: Any) -> str:
    """Return Markdown that embeds a PIL image."""
    return f"![image]({get_img_uri(img)})"


def safe_chat(
    client: OpenAI,
    *,
    max_attempts: int = 5,
    retry_delay: float = 1.5,
    **kwargs: Any,
) -> Any:
    """Call Chat Completions and retry temporary rate-limit failures."""
    for attempt in range(max_attempts):
        try:
            return client.chat.completions.create(**kwargs)
        except openai.RateLimitError:
            if attempt == max_attempts - 1:
                break
            print("Rate limit reached; waiting before retrying...")
            time.sleep(retry_delay)
    raise RuntimeError(
        "The OpenAI rate limit persisted. Wait and try the extraction again."
    )


def analyze_one_page_GPT(
    one_page_text: Sequence[str],
    *,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
) -> list[str]:
    """Extract TLC data from each selected PDF page with an OpenAI model."""
    if client is None:
        client = OpenAI()

    json_style_example = """
    [
      {
        "compound name": "26b",
        "compound IUPAC name": "the given IUPAC name",
        "eluent1": "PE",
        "eluent2": "EA",
        "ratio": "20/1",
        "Rf": "0.6"
      },
      {
        "compound name": "26c",
        "compound IUPAC name": "another IUPAC name",
        "eluent1": "CH2Cl2",
        "eluent2": "np.nan",
        "ratio": "1",
        "Rf": "0.4"
      }
    ]
    """

    print("GPT started to analyze the PDF.")
    answers: list[str] = []

    for index in tqdm(range(len(one_page_text))):
        if index == 0:
            supporting_page_text = (
                "This is the first page, so there is no previous page."
            )
        else:
            supporting_page_text = one_page_text[index - 1]

        if answers:
            clean_answer = (
                answers[-1]
                .strip()
                .removeprefix("```json")
                .removesuffix("```")
                .strip()
            )
            previous_answer = json.loads(clean_answer)
        else:
            previous_answer = []

        system_prompt = f"""
You will be provided with the text of a PDF page.

Check whether each compound on the page meets the following 4 conditions:
1. The text includes a compound with an Rf value specified.
2. The compound with the Rf value must have its IUPAC name explicitly written.
   If the IUPAC name is not found or is ambiguous, completely ignore the
   compound and do not include it in the output.
3. The eluent used to measure the Rf value is specified. It must be either a
   single solvent or a mixture of two solvents.
4. The ratio between the solvents, if it is a mixture, is clearly written.

For every compound that satisfies all four conditions, return a list of
Python-style dictionaries in JSON format, where all values are strings.
Do not guess or fill in a missing field.

The Rf information for a compound appears after the compound name. Associate
the Rf value with the correct preceding compound.

If both TLC and column chromatography solvents are mentioned, use the solvent
composition associated with the Rf measurement. Do not use a purification or
flash-chromatography ratio unless it is explicitly the Rf condition.

Special handling:
- For one eluent, use it as "eluent1", set "eluent2" to "np.nan", and set
  "ratio" to "1".
- For two eluents, assign "eluent1" and "eluent2" in the written order.
- Convert "30% eluent1 in eluent2" to "30/70".
- Keep "eluent1 / eluent2 = 30/1" as "30/1".
- Convert "eluent1 / eluent2 = 20:1" to "20/1".

Output format example:
{json_style_example}

Return only the JSON list. Do not add explanations or formatting.

Do not repeat data already reported on the previous page.
Already reported data: {previous_answer}

Previous page for context:
{supporting_page_text}
"""

        response = safe_chat(
            client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": one_page_text[index]}
                    ],
                },
            ],
            temperature=0,
            top_p=0.1,
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(f"The model returned no content for page {index}.")
        answers.append(content)

    return answers


def collect_rf_pages(
    pdf_file: str | Path,
    *,
    si_dir: str | Path = DEFAULT_SI_DIR,
    poppler_path: str | Path | None = None,
    render_images: bool = False,
) -> tuple[str, list[str], list[list[list[str]]], list[list[str]]]:
    """Collect text and optional image data for PDF pages related to Rf."""
    print("Preprocessing PDF...")
    pdf_path = Path(pdf_file)
    if not pdf_path.is_absolute():
        pdf_path = Path(si_dir) / pdf_path
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    images, united_text, short_text, sentence_windows = pdf_Rf_page_img(
        [pdf_path],
        poppler_path=poppler_path,
        render_images=render_images,
    )
    image_urls = [
        [get_img_uri(page_image) for page_image in paper_images]
        for paper_images in images
    ]
    return united_text, short_text, sentence_windows, image_urls


def normalize_ligatures(value: str) -> str:
    """Normalize typographic ligatures and other compatibility characters."""
    return unicodedata.normalize("NFKC", value)


def most_similar_substring(
    text: str,
    target: Any,
    mode: str = "IUPAC_string",
    starting_index: int = 0,
) -> str | int:
    """Find the fixed-length substring most similar to an IUPAC name."""
    compact_text = normalize_ligatures(re.sub(r"\s+", "", text))
    best_match = ""
    best_index = -1
    highest_ratio = 0.0

    if isinstance(target, str):
        target_length = len(target)
        for index in range(starting_index, len(compact_text) - target_length + 1):
            chunk = compact_text[index : index + target_length]
            ratio = difflib.SequenceMatcher(None, chunk, target).ratio()
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match = chunk
                best_index = index

    if mode == "IUPAC_string":
        return best_match.strip()
    if mode == "IUPAC_index":
        return best_index
    raise ValueError("mode must be 'IUPAC_string' or 'IUPAC_index'")


def most_similar_substring_limited_idx(
    text: str,
    target: Any,
    ending_index: int,
    mode: str = "IUPAC_string",
    starting_index: int = 0,
) -> str | int:
    """Find the most similar substring within a limited index range."""
    compact_text = normalize_ligatures(re.sub(r"\s+", "", text))
    best_match = ""
    best_index = -1
    highest_ratio = 0.0

    if isinstance(target, str):
        target_length = len(target)
        final_start = min(ending_index, len(compact_text) - target_length + 1)
        for index in range(starting_index, max(starting_index, final_start)):
            chunk = compact_text[index : index + target_length]
            ratio = difflib.SequenceMatcher(None, chunk, target).ratio()
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match = chunk
                best_index = index

    if mode == "IUPAC_string":
        return best_match.strip()
    if mode == "IUPAC_index":
        return best_index
    raise ValueError("mode must be 'IUPAC_string' or 'IUPAC_index'")


def false_iupac_compare(
    short_text: Sequence[str],
    data_final: pd.DataFrame,
    false_page_idx: dict[int, Sequence[int]],
) -> None:
    """Print likely source-text matches for IUPAC names requiring review."""
    for page_index, row_indices in false_page_idx.items():
        page_text = short_text[page_index]
        previous_text = short_text[page_index - 1] if page_index > 0 else None
        iupac_names = data_final.loc[
            list(row_indices), "compound IUPAC name"
        ].tolist()

        for position, iupac_name in enumerate(iupac_names):
            row_index = list(row_indices)[position]
            current_match = most_similar_substring(page_text, iupac_name)
            richprint(f"[bold red]index___{row_index}[/bold red]")
            richprint(
                "[bold red]name: "
                f"{data_final.loc[row_index, 'compound IUPAC name']}"
                "[/bold red]"
            )
            normalized_name = (
                re.sub(r"\s+", "", iupac_name)
                if isinstance(iupac_name, str)
                else iupac_name
            )
            print("LLM_output >>>>  ", normalized_name)
            print("TXT_current >>>  ", current_match)
            if previous_text is not None:
                previous_match = most_similar_substring(
                    previous_text, iupac_name
                )
                print(
                    "TXT_previous >>  ",
                    previous_match,
                    "\nUse this when the current-page match is poor.",
                )


def _parse_json_response(response_text: str) -> list[dict[str, str]]:
    clean_data = (
        response_text.strip()
        .removeprefix("```json")
        .removesuffix("```")
        .strip()
    )
    parsed = json.loads(clean_data)
    if not isinstance(parsed, list):
        raise ValueError("The model response must be a JSON list.")
    return parsed


def _highlight_source_text(
    full_text: str,
    iupac_chunks: Iterable[str],
) -> None:
    placeholder_map: dict[str, str] = {}
    unique_chunks = sorted(set(iupac_chunks), key=len, reverse=True)

    for index, chunk in enumerate(unique_chunks):
        pattern = r"\s*".join(re.escape(character) for character in chunk)
        placeholder = f"__IUPAC_PLACEHOLDER_{index}__"

        def replace_match(match: re.Match[str], key: str = placeholder) -> str:
            placeholder_map[key] = match.group(1)
            return key

        full_text = re.sub(
            f"({pattern})",
            replace_match,
            full_text,
            flags=re.IGNORECASE,
        )

    colored_lines = [
        f"__GREEN_START__{line}__GREEN_END__"
        if "Rf" in line or "RF" in line
        else line
        for line in full_text.split("\n")
    ]
    colored_text = "\n".join(colored_lines)

    for placeholder, original_text in placeholder_map.items():
        colored_text = colored_text.replace(
            placeholder,
            f"[bold orange1]{original_text}[/bold orange1]",
        )
    colored_text = colored_text.replace(
        "__GREEN_START__", "[bold green1]"
    ).replace("__GREEN_END__", "[/bold green1]")
    Console().print(colored_text, highlight=False)


def extract_data_GPT(
    pdf_file: str | Path,
    *,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
    si_dir: str | Path = DEFAULT_SI_DIR,
    poppler_path: str | Path | None = None,
    review: bool = True,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]] | None:
    """Run extraction and display page-by-page review information."""
    _, short_text, _, _ = collect_rf_pages(
        pdf_file,
        si_dir=si_dir,
        poppler_path=poppler_path,
        render_images=False,
    )
    if not short_text:
        print("Cannot detect Rf pages.")
        return None

    extracted_responses = analyze_one_page_GPT(
        short_text,
        client=client,
        model=model,
    )
    false_page_idx: dict[int, np.ndarray] = {}
    whole_data: list[dict[str, str]] = []
    dataframe_index_start = 0

    for page_index, response in enumerate(extracted_responses):
        page_data = _parse_json_response(response)
        if not page_data:
            print(f"No Rf data on selected page {page_index}.")
            continue

        page_frame = pd.DataFrame(page_data)
        missing_columns = set(EXTRACTED_COLUMNS) - set(page_frame.columns)
        if missing_columns:
            raise ValueError(
                f"Page {page_index} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        richprint(
            f"[bold bright_red]Page "
            f"---#################################---{page_index}"
            f"[/bold bright_red]"
        )
        page_frame.index = np.arange(
            dataframe_index_start,
            dataframe_index_start + len(page_frame),
        )
        dataframe_index_start += len(page_frame)
        display(
            page_frame[
                [
                    "compound IUPAC name",
                    "eluent1",
                    "eluent2",
                    "ratio",
                    "Rf",
                ]
            ]
        )

        compact_page = normalize_ligatures(
            re.sub(r"\s+", "", short_text[page_index]).lower()
        )
        page_iupac_indices = [
            compact_page.find(
                re.sub(
                    r"\s+",
                    "",
                    normalize_ligatures(str(iupac_name)),
                ).lower()
            )
            for iupac_name in page_frame["compound IUPAC name"]
        ]
        review_indices = page_iupac_indices.copy()

        if review_indices and review_indices[0] == -1 and page_index > 0:
            compact_previous = normalize_ligatures(
                re.sub(r"\s+", "", short_text[page_index - 1]).lower()
            )
            previous_indices = [
                compact_previous.find(
                    re.sub(
                        r"\s+",
                        "",
                        normalize_ligatures(str(iupac_name)),
                    ).lower()
                )
                for iupac_name in page_frame["compound IUPAC name"]
            ]
            for index in np.flatnonzero(np.array(previous_indices) != -1):
                shifted_index = previous_indices[index] - (
                    len(compact_previous) + 100
                )
                review_indices[index] = shifted_index
                page_iupac_indices[index] = shifted_index

        if -1 in review_indices:
            missing_positions = np.flatnonzero(np.array(review_indices) == -1)
            missing_rows = missing_positions + (
                dataframe_index_start - len(page_frame)
            )
            print(f"Wrong IUPAC in this page: {missing_rows}")
            false_page_idx[page_index] = missing_rows

            for position in missing_positions:
                query = page_frame.iloc[position]["compound IUPAC name"]
                left_candidates = [
                    value for value in review_indices[:position] if value != -1
                ]
                right_candidates = [
                    value
                    for value in review_indices[position + 1 :]
                    if value != -1
                ]
                left = max(left_candidates, default=0)
                right = min(
                    right_candidates,
                    default=len(compact_page) - len(str(query)) + 1,
                )
                right = max(right - len(str(query)), left + 1)
                review_indices[position] = int(
                    most_similar_substring_limited_idx(
                        compact_page,
                        query,
                        ending_index=right,
                        mode="IUPAC_index",
                        starting_index=max(left, 0),
                    )
                )

        if review:
            include_previous = any(value < 0 for value in review_indices)
            if include_previous and page_index > 0:
                full_text = (
                    short_text[page_index - 1]
                    + "\n\n--- [Current Page] ---\n\n"
                    + short_text[page_index]
                )
            else:
                full_text = short_text[page_index]

            compact_current = normalize_ligatures(
                re.sub(r"\s+", "", short_text[page_index])
            )
            compact_previous = (
                normalize_ligatures(
                    re.sub(r"\s+", "", short_text[page_index - 1])
                )
                if page_index > 0
                else ""
            )
            chunks: list[str] = []
            for position, iupac_index in enumerate(review_indices):
                iupac_name = page_frame.iloc[position]["compound IUPAC name"]
                if pd.isna(iupac_name):
                    continue
                name_length = len(str(iupac_name))
                if iupac_index < 0:
                    original_index = max(
                        iupac_index + len(compact_previous) + 100,
                        0,
                    )
                    label = compact_previous[
                        original_index : original_index + name_length
                    ]
                else:
                    label = compact_current[
                        iupac_index : iupac_index + name_length
                    ]
                if label:
                    chunks.append(label)
            _highlight_source_text(full_text, chunks)

        whole_data.extend(page_data)

    data_final = pd.DataFrame(whole_data)
    if data_final.empty:
        print("No data.")
        return None

    data_final = data_final.replace("np.nan", np.nan)
    data_final.index = range(len(data_final))

    if not false_page_idx:
        print("Every IUPAC name was found in the source text.")
    else:
        richprint("[bold red]IUPAC names requiring review[/bold red]")
        richprint(false_page_idx)
        false_iupac_compare(short_text, data_final, false_page_idx)

    return data_final, false_page_idx


def suggested_deletion_indices(
    false_page_idx: dict[int, Sequence[int]],
) -> list[int]:
    """Flatten the row indices flagged during source-text verification."""
    return sorted(
        {
            int(index)
            for page_indices in false_page_idx.values()
            for index in page_indices
        }
    )


def remove_rows(
    dataframe: pd.DataFrame,
    row_indices: Iterable[int],
) -> pd.DataFrame:
    """Return a copy with selected rows removed and the index reset."""
    result = dataframe.drop(index=list(row_indices), errors="ignore").copy()
    return result.reset_index(drop=True)


def clean_extracted_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove incomplete records required by the TLC dataset."""
    required = ["compound IUPAC name", "eluent1", "ratio", "Rf"]
    result = dataframe.dropna(subset=required).copy()
    return result.reset_index(drop=True)


def deduplicate_extracted_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create the original composite data_ID and remove duplicate records."""
    result = dataframe.copy()
    id_columns = [
        "compound IUPAC name",
        "eluent1",
        "eluent2",
        "ratio",
        "Rf",
    ]
    result["data_ID"] = result[id_columns].astype(str).agg("_".join, axis=1)
    aggregations = {
        column: "first"
        for column in EXTRACTED_COLUMNS
        if column != "data_ID"
    }
    return (
        result.groupby("data_ID", as_index=False)
        .agg(aggregations)
        .reset_index(drop=True)
    )


def iupac_to_smiles_local_opsin(
    iupac_name: str,
    *,
    jar_path: str | Path = DEFAULT_OPSIN_JAR,
) -> str | None:
    """Convert one IUPAC name to SMILES with the local OPSIN JAR."""
    resolved_jar = Path(jar_path)
    if not resolved_jar.is_file():
        raise FileNotFoundError(
            f"OPSIN JAR not found: {resolved_jar}. "
            "Run `python download_opsin.py` first."
        )

    result = subprocess.run(
        ["java", "-jar", str(resolved_jar)],
        input=iupac_name,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"OPSIN failed: {result.stderr.strip()}")
    return result.stdout.strip() or None


def convert_iupac_to_smiles(
    dataframe: pd.DataFrame,
    *,
    jar_path: str | Path = DEFAULT_OPSIN_JAR,
) -> pd.DataFrame:
    """Convert the IUPAC-name column to SMILES and discard failed rows."""
    smiles_values = [
        iupac_to_smiles_local_opsin(str(iupac), jar_path=jar_path)
        if pd.notna(iupac) and str(iupac).strip()
        else None
        for iupac in tqdm(dataframe["compound IUPAC name"])
    ]
    result = dataframe.copy()
    result["compound IUPAC name"] = smiles_values
    result = result.rename(columns={"compound IUPAC name": "Smiles"})
    return result.dropna(subset=["Smiles"]).reset_index(drop=True)


def prepare_export_data(
    dataframe: pd.DataFrame,
    *,
    title: str,
    doi: str,
    publish: str,
) -> pd.DataFrame:
    """Attach paper metadata to an extracted TLC dataframe."""
    result = dataframe.dropna(subset=["Smiles"]).copy()
    result["title"] = title
    result["doi"] = doi
    result["publish"] = publish
    return result.reset_index(drop=True)


def list_datatable_files(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    """List numbered datatable CSV files in numeric order."""
    directory = Path(output_dir)
    if not directory.exists():
        return []

    numbered_files: list[tuple[int, Path]] = []
    for path in directory.glob("datatable*.csv"):
        match = re.fullmatch(r"datatable(\d+)\.csv", path.name)
        if match:
            numbered_files.append((int(match.group(1)), path))
    return [path for _, path in sorted(numbered_files)]


def get_last_saved_metadata(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[str | None, str | None]:
    """Return the title and DOI in the latest numbered datatable."""
    files = list_datatable_files(output_dir)
    if not files:
        return None, None
    last_file = pd.read_csv(files[-1])
    if last_file.empty:
        return None, None
    title = str(last_file.iloc[0]["title"]) if "title" in last_file else None
    doi = str(last_file.iloc[0]["doi"]) if "doi" in last_file else None
    return title, doi


def save_next_datatable(
    dataframe: pd.DataFrame,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    prevent_duplicate_doi: bool = True,
) -> Path:
    """Save data using the next datatable number, with a DOI duplicate guard."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    files = list_datatable_files(directory)

    if prevent_duplicate_doi and files and "doi" in dataframe:
        _, last_doi = get_last_saved_metadata(directory)
        current_doi = (
            str(dataframe.iloc[0]["doi"]) if not dataframe.empty else None
        )
        if current_doi and current_doi == last_doi:
            raise ValueError(
                "The DOI matches the latest saved datatable. "
                "Review the metadata before saving."
            )

    next_index = 1
    if files:
        match = re.fullmatch(r"datatable(\d+)\.csv", files[-1].name)
        if match:
            next_index = int(match.group(1)) + 1
    destination = directory / f"datatable{next_index}.csv"
    dataframe.to_csv(destination, index=False)
    return destination


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_OPSIN_JAR",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SI_DIR",
    "analyze_one_page_GPT",
    "clean_extracted_data",
    "collect_rf_pages",
    "convert_iupac_to_smiles",
    "deduplicate_extracted_data",
    "extract_data_GPT",
    "false_iupac_compare",
    "get_img_markdown",
    "get_img_uri",
    "get_last_saved_metadata",
    "iupac_to_smiles_local_opsin",
    "list_datatable_files",
    "make_buffer_page",
    "most_similar_substring",
    "most_similar_substring_limited_idx",
    "normalize_ligatures",
    "pdf_Rf_page_img",
    "prepare_export_data",
    "remove_rows",
    "safe_chat",
    "save_next_datatable",
    "suggested_deletion_indices",
]

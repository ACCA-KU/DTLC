# LLM Data Extractor

This directory contains a notebook workflow for extracting TLC retention
factor (Rf) data from research papers with the OpenAI API. It uses OPSIN to
convert IUPAC chemical names to SMILES.

## Files

- `GPT_paper_extractor.py`: reusable extraction and post-processing functions
- `GPT_paper_extract.ipynb`: step-by-step notebook used to run the workflow
- `download_opsin.py`: downloads the required OPSIN command-line JAR
- `SI/`: place supporting-information PDF files in this directory

## Requirements

- Python 3
- Java 8 or later
- An OpenAI API key

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Set the API key through the `OPENAI_API_KEY` environment variable, or enter it
when prompted by the notebook. Do not store API keys in source files.

## Running the extraction

1. Put the supporting-information PDF in `SI/`.
2. Open `GPT_paper_extract.ipynb`.
3. Edit the PDF filename and publication metadata.
4. Run the cells in order.
5. Review the colored source text and the extracted tables before saving.

## OPSIN installation

Download OPSIN 2.8.0 from its official GitHub release by running the following
command from this directory:

```bash
python download_opsin.py
```

Confirm that OPSIN works:

```bash
java -jar opsin-cli-2.8.0-jar-with-dependencies.jar -h
```

The downloaded JAR must remain in the same directory as
`GPT_paper_extractor.py`.

## OPSIN attribution

OPSIN is developed by Daniel Lowe and contributors and is distributed under
the MIT License. The OPSIN JAR is downloaded directly from the official
release and is not maintained by this repository.

- Source: https://github.com/dan2097/opsin
- Version 2.8.0 release: https://github.com/dan2097/opsin/releases/tag/2.8.0
- License: https://github.com/dan2097/opsin/blob/master/LICENSE.txt

If you use OPSIN in academic work, please cite:

> Lowe, D. M.; Corbett, P. T.; Murray-Rust, P.; Glen, R. C.  
> Chemical Name to Structure: OPSIN, an Open Source Solution.  
> *Journal of Chemical Information and Modeling* **2011**, *51*, 739-753.  
> https://doi.org/10.1021/ci100384d

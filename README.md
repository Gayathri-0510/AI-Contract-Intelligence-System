# AI Contract Analyzer

This project is a `Streamlit` web application that analyzes contract text using a trained deep learning model.

It is designed to:
- classify the input text as `Contradiction`, `Entailment`, or `Neutral`
- estimate a confidence score for the prediction
- highlight commonly used terms in the text
- flag potentially risky clauses based on keyword heuristics
- generate a downloadable text report

The UI is branded in the app as `Lex·AI Contract Intelligence System`.

## What This Project Does

The app accepts contract content in one of two ways:
- upload a `.txt` file
- paste plain text directly into the interface

After you run the analysis, the app:
- preprocesses the text
- converts it into sequences using a saved tokenizer
- sends the padded sequence to a TensorFlow/Keras model
- shows the predicted label and confidence
- displays supporting charts and statistics
- creates a downloadable summary report

In addition to the model prediction, the app also provides:
- readability and document statistics
- top key terms after stopword removal
- sentence-level risk tagging using simple keyword rules
- simulated visual layers such as attention and positional encoding displays

## Main Files

- `app.py` - main Streamlit application
- `attention_model.h5` - trained model used for inference
- `tokenizer.pkl` - tokenizer used to prepare text input
- `requirements.txt` - Python dependencies

## Tech Stack

- Python
- Streamlit
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn

## How It Works

1. The user uploads or pastes contract text.
2. The app cleans the text and computes basic statistics.
3. The tokenizer converts text into token sequences.
4. The sequences are padded to a fixed length.
5. The model predicts one of three classes:
   - `Contradiction`
   - `Entailment`
   - `Neutral`
6. The app shows charts, risk indicators, and a downloadable report.

## Setup

Make sure these files are present in the project root:
- `app.py`
- `attention_model.h5`
- `tokenizer.pkl`
- `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run The App

```bash
streamlit run app.py
```

After starting, Streamlit will open the app in your browser.

## Input Support

- Supported upload format: `.txt`
- You can also paste text directly into the text area

## Output

The app produces:
- predicted contract classification
- confidence score
- token frequency chart
- risk distribution summary
- attention and positional encoding visualizations
- downloadable report as `lexai_contract_report.txt`

## Important Note

If the model or tokenizer cannot be loaded, the app does not fully stop.
Instead, it falls back to a simulated prediction mode using randomly generated probabilities.

That means:
- the interface still works
- charts still render
- results are not real model predictions in fallback mode

## Limitations

- upload support is limited to plain text files
- there are no tests or deployment files included in this project
- dependency versions are not pinned in `requirements.txt`
- some visualizations are illustrative rather than derived from real model attention weights

## Suggested Future Improvements

- add support for `PDF` and `DOCX` contracts
- pin dependency versions
- add automated tests
- save reports in structured formats like `CSV` or `JSON`
- connect risk detection to a stronger legal clause analysis pipeline

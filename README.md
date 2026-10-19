# CGMacros: Multimodal AI for Multi-Horizon Glucose Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> **New:** [`leakfree-benchmark/`](leakfree-benchmark/) holds the leak-free, participant-held-out benchmark and the audit of the released CGMacros CGM series (the released 1-minute series interpolates 15-minute readings and leaks the next reading). The code in `src/` below is the earlier multimodal framework and is unchanged.

A multimodal machine learning framework that forecasts post-meal glucose spikes
across **multiple time horizons (1h, 3h, 6h)**, combining continuous glucose
monitoring (CGM) signals, meal macronutrient data, clinical biomarkers, and
AI-derived food image embeddings.

## Table of Contents

- [Overview](#overview)
- [Approach](#approach)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Dataset](#dataset)
- [License](#license)

## Overview

Predicting how a person's blood glucose will respond to a meal is a core problem
in personalized nutrition and diabetes management. Single-modality models
(CGM history alone, or nutrition facts alone) miss information that's only
visible when signals are combined. This project builds a **multimodal** pipeline
that fuses:

- **CGM time-series** — recent glucose trajectory leading up to a meal
- **Nutritional features** — macronutrient breakdown of the meal
- **Clinical biomarkers** — patient-level health indicators
- **Food image embeddings** — visual features extracted from meal photos

to predict glucose response at three forecast horizons (1, 3, and 6 hours
post-meal).

## Approach

The modeling code (`src/model.py`) implements and compares three architectures:

- **LSTM** — sequence model over the CGM time-series
- **Transformer** — self-attention model for capturing longer-range dependencies
- **Ensemble** — averages predictions across multiple base models

Data ingestion and preparation are handled by `src/data_loader.py` (CSV/Excel/JSON
loading) and `src/preprocessing.py`, with `src/evaluate.py` providing evaluation
utilities and `src/main.py` as the pipeline entry point.

## Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch, TensorFlow, Keras |
| Data Processing | pandas, NumPy, SciPy |
| Classical ML | scikit-learn |
| Visualization | Matplotlib, Seaborn, Plotly |
| Tooling | Jupyter, pytest, black, flake8, mypy |

See [`requirements.txt`](requirements.txt) for the full dependency list.

## Project Structure

```
.
├── notebooks/
│   └── CGMacros_Modeling.ipynb   # End-to-end exploration, modeling & evaluation
├── src/
│   ├── data_loader.py            # CSV / Excel / JSON ingestion
│   ├── preprocessing.py          # Feature preparation
│   ├── model.py                  # LSTM, Transformer & Ensemble architectures
│   ├── evaluate.py                # Evaluation utilities
│   └── main.py                    # Pipeline entry point
├── docs/
│   ├── Report.pdf                # Full write-up of methodology & findings
│   └── DATASET.md                # Dataset source & access notes
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/Pulak-PSR/CGMacros-Multimodal-AI-for-Multiple-Horizon-Glucose-Prediction.git
cd CGMacros-Multimodal-AI-for-Multiple-Horizon-Glucose-Prediction
pip install -r requirements.txt
```

## Usage

Open the main notebook for the full walkthrough:

```bash
jupyter notebook notebooks/CGMacros_Modeling.ipynb
```

Or run the pipeline script directly:

```bash
python src/main.py
```

## Dataset

This project uses the public **CGMacros** dataset. See [`docs/DATASET.md`](docs/DATASET.md)
for the source link and setup notes — the raw data is not included in this repo.

## License

Released under the [MIT License](LICENSE).

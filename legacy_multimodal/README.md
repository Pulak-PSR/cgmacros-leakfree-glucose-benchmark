# Legacy: CGMacros multimodal multi-horizon glucose prediction (superseded)

> **Superseded. Do not rely on the results of this folder.**
> This is the original MSc dissertation code (LSTM, Transformer and ensemble models on CGM history, meal macronutrients, clinical markers and meal-photo embeddings). It was written before the CGM series was audited. The audit in the repository root found that the released 1-minute CGM series is a linear interpolation of 15-minute readings, so features computed at intermediate minutes contain the next reading. Separately, the dissertation's headline accuracy was not reproduced under a corrected, participant-held-out evaluation. The corrected, participant-held-out benchmark is in the repository root (see the top-level [README](../README.md)).

The files are kept unchanged for the record (moved here with their git history):

```
legacy_multimodal/
├── notebooks/CGMacros_Modeling.ipynb   # original exploration, modeling and evaluation
├── src/                                # data_loader, preprocessing, model, evaluate, main
├── docs/Report.pdf, docs/DATASET.md    # original write-up and dataset note
├── requirements.txt                    # original (unpinned) dependencies
├── LICENSE                             # MIT
└── gitignore.legacy                    # original .gitignore
```

Original description: a multimodal framework forecasting post-meal glucose at 1, 3 and 6 hours by fusing CGM history, meal macronutrients, clinical biomarkers and AI-derived food-image embeddings.

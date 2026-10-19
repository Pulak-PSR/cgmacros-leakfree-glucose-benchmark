# Changelog

## 1.0.0 (2026-10-19)

* Repository renamed from `CGMacros-Multimodal-AI-for-Multiple-Horizon-Glucose-Prediction` to `cgmacros-leakfree-glucose-benchmark` and restructured around the leak-free benchmark and CGM series audit.
* Added: `cgmacros_bench/` library, scripts for the audit, leak experiment, benchmark, secondary experiments, figures and result verification, `expected_results/`, `docs/`, CI, issue template, `CITATION.cff`.
* The original dissertation code (LSTM, Transformer, ensemble on CGM history, nutrition, clinical and image features) moved unchanged to `legacy_multimodal/` and is marked superseded.

# Getting the data

The benchmark uses the public **CGMacros** dataset (v1.0.0), hosted on PhysioNet: https://physionet.org/content/cgmacros/1.0.0/

1. Open the project page, read its licence (CC BY-NC-SA 4.0: attribution, non-commercial, share-alike), and download the files.
2. Unzip. The folder you pass to the scripts is the one containing `CGMacros_dateshifted365/CGMacros/CGMacros-001, ...` and `bio.csv`.
3. Either set `CGMACROS_DIR=/path/to/that/folder` or pass `--data-dir` to each script.

The data are **not** included in this repository and must not be committed: `.gitignore` excludes `data/`, `cache/`, `results/`, `*.npz` and `*.zip`, and CI fails if data-like files are tracked.

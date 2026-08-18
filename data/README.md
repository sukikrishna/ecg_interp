# Datasets

Both datasets below are **fully open access on PhysioNet — no account, login, data-use
agreement, or CITI training required.** Anyone can run the download scripts directly.

## PTB-XL (primary dataset)

- Source: https://physionet.org/content/ptb-xl/1.0.3/
- License: Creative Commons Attribution 4.0 (CC BY 4.0)
- Size: ~1.7GB zipped / ~3GB on disk
- 21,801 clinical 12-lead ECGs from 18,869 patients, with SCP-ECG statement labels that map onto
  the 5 target concepts (atrial fibrillation, bundle branch block, normal rhythm, left
  ventricular hypertrophy, myocardial infarction) via `ptbxl_database.csv` + `scp_statements.csv`.

Download:
```bash
bash scripts/download_ptbxl.sh
```
which fetches into `data/raw/ptb-xl/` (gitignored — re-run the script rather than committing it).

**Citation** (please cite both when using this data):
> Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F.I., Samek, W.,
> Schaeffter, T. (2020). PTB-XL: A Large Publicly Available ECG Dataset. *Scientific Data*.

> Wagner, P., et al. (2022). PTB-XL, a large publicly available electrocardiography dataset
> (version 1.0.3). PhysioNet.

> Pollard, T., Moody, B.E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R.G.,
> Heldt, T. (2026). PhysioNet as a global platform for biomedical research.

## PTB Diagnostic ECG Database (secondary — smaller, older, optional)

- Source: https://physionet.org/content/ptbdb/1.0.0/
- License: Open Data Commons Attribution License v1.0
- Size: ~1.7GB
- 549 records from 290 subjects — mainly useful as a secondary/held-out check, not the primary
  training set (much smaller and less granularly labeled than PTB-XL).

Download:
```bash
bash scripts/download_ptbdb.sh
```
which fetches into `data/raw/ptbdb/` (gitignored).

**Citation:**
> Bousseljot R, Kreiseler D, Schnabel A. Nutzung der EKG-Signaldatenbank CARDIODAT der PTB über
> das Internet. *Biomedizinische Technik*, Band 40, Ergänzungsband 1 (1995), S 317.
>
> Dataset DOI: https://doi.org/10.13026/C28C71

## Looking ahead: EHR data

If/when this project extends to structured-EHR foundation models (see the "longer-term
direction" note in [../docs/research-plan.md](../docs/research-plan.md)), datasets like MIMIC-IV
are **not** open access the way PTB-XL/PTB are — they require a PhysioNet **credentialed access**
application (identity verification + a short CITI "Data or Specimens Only Research" training
course + a signed data use agreement), which only the account holder can complete. That's not
needed for the current ECG scope, but flagging it now since it's the kind of manual step that
takes a few days to clear and is worth starting early once that phase begins.

# Uplift Modeling on the Hillstrom Email Dataset

Estimating who should receive a marketing email, not just who is likely to buy.

## 1. Business question

A retailer ran a randomized email campaign: each customer was assigned to receive a
Men's-catalog email, a Women's-catalog email, or no email at all. The naive question --
"who is likely to visit/convert?" -- rewards customers who would have bought anyway and
says nothing about whether the email caused anything. This project asks the causal
question instead: **for which customers does sending the email actually change their
behavior, and how much?**

That's an uplift (heterogeneous treatment effect) modeling problem, not a plain
classification problem. The deliverable is a targeting rule -- send to the top X% of
customers by predicted incremental effect -- backed by held-out evaluation, not just
in-sample accuracy.

## 2. Dataset

Dataset provided by Kevin Hillstrom / MineThatData for research and educational use; not redistributed in this repository — fetched at runtime via scikit-uplift's fetch_hillstrom.
[Kevin Hillstrom's MineThatData e-mail analytics dataset](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html):
a real randomized controlled experiment covering **~64,000 customers**, each assigned to
one of three arms (`Womens E-Mail`, `Mens E-Mail`, `No E-Mail`). Pre-treatment covariates
include recency, historical spend and its binned segment, prior purchase categories
(mens/womens), account tenure (`newbie`), zip-code region, and acquisition channel.
Outcomes are `visit`, `conversion`, and `spend`, tracked for two weeks post-campaign. The
raw CSV is fetched automatically via `scikit-uplift`'s `fetch_hillstrom` the first time
`load_hillstrom` runs (see Setup below) -- it is not checked into this repo.

## 3. Repository structure

```
uplift-modeling-hillstrom/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                       # downloaded Hillstrom CSV (not committed)
│   └── processed/                 # train/val/test parquet + manifest + dashboard artifact
├── notebooks/
│   ├── 01_eda.ipynb               # randomization check, covariate balance, outcome rates
│   ├── 02_data_prep.ipynb         # encoding, split, persist to data/processed/
│   ├── 03_uplift_models.ipynb     # S-/T-/X-Learner, baselines, TARNet, ensemble (val-based)
│   ├── 04_evaluation.ipynb        # held-out test metrics, the project's authoritative numbers
│   └── 05_heterogeneity_shap.ipynb  # SHAP decomposition of the Womens X-Learner's uplift
├── src/
│   ├── data_prep.py                # loading, balance checks, encoding, splitting
│   ├── models.py                   # base learners, hand-rolled XLearner, TARNet, SHAP decomposition
│   └── evaluation.py                # bootstrap-CI Qini/uplift@k, Qini curves, targeting impact
├── app/
│    build_dashboard_artifacts.py  # precomputes fits/CIs/SHAP so the dashboard stays responsive
│   ├── dashboard.py                # Streamlit app: balance check, model comparison, targeting simulator, SHAP
└── docs/
    └── methodology.md              # full methodological write-up, one section per pipeline stage
```

Notebooks are for exploration and narrative only -- they import from `src/` rather than
redefining logic inline, so every notebook and the dashboard share exactly the same
data-loading, modeling, and evaluation code.

## 4. Setup

This project assumes `conda` and installs project-specific packages inside notebook
cells via `%pip install` (not `!pip install` or a separate terminal `pip`), so each
notebook can be run top-to-bottom on a fresh environment without an implicit setup step
elsewhere.

```bash
# 1. Create and activate a clean environment
conda create -n uplift-hillstrom python=3.11 -y
conda activate uplift-hillstrom

# 2. Install Jupyter itself (not a project dependency, so it's not in requirements.txt)
conda install -c conda-forge jupyterlab -y

# 3. Install the project's own dependencies
pip install -r requirements.txt
```

With this environment active, every notebook's first cell (`%pip install ...`) will
find its dependencies already satisfied and skip straight through -- the `%pip install`
cells exist so each notebook is independently runnable even outside this exact setup
flow, not as a substitute for it.

## 5. Running the project

**Notebooks**, in order, from the repository root (each reads/writes paths relative to
the root, e.g. `data/processed/`):

```bash
jupyter lab
# run 01_eda.ipynb, then 02_data_prep.ipynb, then 03_uplift_models.ipynb,
# then 04_evaluation.ipynb, then 05_heterogeneity_shap.ipynb
```

`02_data_prep.ipynb` must run before any later notebook -- it produces
`data/processed/{train,val,test}.parquet` and `feature_manifest.json`, which everything
downstream loads.

**Dashboard.** The Streamlit app reads precomputed artifacts rather than refitting
models on every widget interaction, so build the artifact bundle once first:

```bash
python scripts/build_dashboard_artifacts.py   # writes data/processed/dashboard_artifacts.joblib
streamlit run app/dashboard.py
```

Re-run `build_dashboard_artifacts.py` whenever `data/processed/` changes.

## 6. Methodology summary

Full reasoning, including every assumption check and the rationale behind each modeling
choice, lives in [`docs/methodology.md`](docs/methodology.md) and
[`docs/decisions_log.md`](docs/decisions_log.md). In brief:

- **Causal validity comes first.** Before any modeling, covariate balance across arms is
  checked via standardized mean differences (all well under the 0.1 flagging threshold),
  and the design matrix is checked for rank deficiency and multicollinearity. This is
  what licenses treating the fitted effects as causal rather than merely predictive --
  ignorability holds by construction here because treatment was randomized, and SUTVA is
  assumed to hold (customers' outcomes don't depend on which arm other customers were
  assigned to).
- **Two binary comparisons, not one three-arm model.** `scikit-uplift`'s meta-learners
  expect a binary treatment, and pooling the two email arms would average over genuinely
  different interventions. The analysis runs as two independent RCTs: Mens vs No E-Mail,
  and Womens vs No E-Mail.
- **`visit`, not `conversion`, is the modeling target** -- `conversion` is under 1% of
  customers, too rare at this sample size for reliable heterogeneous-effect estimation.
- **Models compared:** S-Learner, T-Learner, and a hand-rolled X-Learner (all GBM-based,
  plus a logistic-regression S-Learner for contrast), extended with Decision Tree and
  Random Forest baselines, an MLP, TARNet (a neural architecture purpose-built for
  treatment-effect estimation), and a top-3 uplift-score ensemble.
- **Evaluation is entirely held-out.** `test.parquet` is touched exactly once, in
  `04_evaluation.ipynb`, to produce Qini AUC and uplift@k with bootstrap confidence
  intervals -- point estimates alone aren't trusted at this sample size.
- **Heterogeneity is explained, not just measured.** `05_heterogeneity_shap.ipynb` uses
  an exact SHAP decomposition (enabled by the X-Learner's architecture) to show which
  features drive predicted uplift up or down for the one model this evaluation actually
  supports deploying.

## 7. Key results

| Arm | Best model | Qini AUC (95% CI) | Deploy? |
|---|---|---|---|
| Womens E-Mail | X-Learner (GBM) | 0.076 (CI excludes zero) | Yes -- statistically tied with S-Learner (GBM) at 0.072; both lead |
| Mens E-Mail | S-Learner (GBM) | 0.012 (CI includes zero) | No -- every model's ranking is statistically indistinguishable from random |

- For the **Womens** arm, targeting the top 10-30% of customers by predicted uplift beats
  random targeting at every tested budget, regardless of which of the two leading models
  is used.
- For the **Mens** arm, the evidence does not support deploying a targeting model at all
  on the current data and feature set -- every candidate's confidence interval on Qini
  AUC includes zero.
- SHAP-based drivers of predicted uplift for the Womens X-Learner are detailed in
  `docs/methodology.md`'s Business Interpretation section once `05_heterogeneity_shap.ipynb`
  has been run against the current `data/processed/` split.

See `docs/methodology.md` for the full evaluation protocol (bootstrap CIs, business-framed
targeting impact, and the assumptions each figure depends on) and `docs/decisions_log.md`
for the reasoning behind each modeling choice.

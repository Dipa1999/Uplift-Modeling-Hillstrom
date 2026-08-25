# Methodology

## Problem Framing

This project does not ask "who is most likely to convert?" -- a plain prediction question
that would just reward customers who were going to buy anyway. Instead it asks: **who
should receive the email campaign to maximize the incremental (causal) effect of the
campaign?** This is an uplift modeling problem: the target is treatment-effect
heterogeneity across customers, not an outcome probability.

The Hillstrom dataset is a real randomized experiment with three arms (`Womens E-Mail` /
`Mens E-Mail` / `No E-Mail`), so a causal interpretation of the effect estimates produced
later is licensed *if* randomization held up in practice and the covariate set is
well-behaved. Both are checked explicitly below, before any modeling begins.

## Data & Randomization Check

Across the pre-treatment covariates checked (recency, history, purchase-category flags,
newbie status, zip code region, channel), standardized mean differences between each email
segment and the no-email control were uniformly small (max |SMD| ≈ 0.014, well under the
0.1 threshold), and no covariate/group combination was flagged as imbalanced --
consistent with randomization having held as designed. Structurally, `history_segment` was
confirmed to be a clean, non-overlapping binning of `history` (7 bins, no overlapping
min/max ranges), so the two are never used together in a linear base learner. The one-hot
encoded covariate set (`zip_code`, `channel`, with an intercept) is rank-deficient by the
dummy variable trap when `drop_first=False` (9 columns, rank 7) and full rank once
`drop_first=True` is applied (7 columns, rank 7); VIF values were all comfortably below 5
(max ≈ 3.05, for `channel_Phone`), indicating no material near-collinearity to address
before modeling. Raw outcome rates show both email segments outperforming the no-email
control on `visit` and `conversion`, as expected for a validated experimental dataset.
Given randomization/ignorability and SUTVA both look reasonable, and the covariate
structure is sound, we proceed to uplift modeling with reasonable confidence that
estimated effects reflect the campaign's causal impact rather than confounding or
modeling artifacts.

## Data Preparation

Train/validation/test splits (60/20/20) are drawn with `stratified_train_val_test_split`
in `src/data_prep.py`, stratified only on the 3-arm `segment` column. This preserves the
property that matters most given this is a randomized experiment: every split is itself
a valid, self-contained comparison across arms. Outcome rates (`visit`, `conversion`) are
not explicitly stratified on, since at n of about 64,000 they land close to the population
rate by the same law-of-large-numbers logic used to read the SMD balance check, and
compounding the stratification key would add complexity without a measurable benefit.

Covariates are encoded once, on the full dataset, before splitting: `zip_code` and
`channel` are one-hot encoded with `drop_first=True` (matching the full-rank design
matrix confirmed in `01_eda.ipynb`), and `history_segment` is dropped, since it was
confirmed to be a clean, non-overlapping binning of the continuous `history` column and
therefore adds no information for any model class used downstream. Encoding before
splitting is safe here because it's a fixed, deterministic transform of columns whose
categories were already locked in by `load_hillstrom`'s categorical dtype cast -- unlike
scaling or imputation, it does not "learn" anything from the data, so it cannot leak
information across splits. No interaction terms, scaling, or other feature engineering
are applied at this stage: those are model-class-specific choices (e.g. scaling matters
for a linear base learner but not a tree-based one, and must be fit on training data only
within each modeling pipeline/fold to avoid leakage), so they belong in the modeling
notebook, not baked once into data saved to disk.

The 3-arm `segment` column is preserved as-is through this notebook, not collapsed into a
single binary treatment indicator. scikit-uplift's meta-learners (S-/T-/X-Learner, per
`decisions_log.md`) expect a binary treatment, but this experiment has two treatment arms
(Mens E-Mail, Womens E-Mail) plus one control; `make_binary_treatment` in
`src/data_prep.py` filters to one arm + control and produces a binary `treatment` column
on demand, so that choice is made in the modeling notebook once per analysis rather than
fixed here.

Processed splits are saved as `data/processed/{train,val,test}.parquet` -- separate files
rather than one file with a split-indicator column, so the file boundary itself prevents
accidentally loading validation or test rows into training -- plus a
`data/processed/feature_manifest.json` recording feature/outcome/treatment column names
and per-split segment proportions, so the modeling notebook doesn't need to hardcode
column lists.

## Modeling Approach

**Treatment reduction.** `segment` has three levels but scikit-uplift's meta-learners
expect a binary treatment. Rather than pooling both email arms (which would average over
genuinely different interventions) or dropping one arm, the analysis runs as **two
parallel binary comparisons**: Mens E-Mail vs No E-Mail, and Womens E-Mail vs No E-Mail.
Each is a valid, self-contained RCT on its own, since any two-arm subset of a randomized
three-arm design is itself a clean two-arm RCT. Caveat: both comparisons share the same
`No E-Mail` control, so their sampling noise is correlated -- a direct claim like "uplift
is higher for Mens than Womens" would need to account for that; no such claim is made.

**Target variable: `visit`, not `conversion`.** `conversion` is under 1% of customers
(01_eda.ipynb), too rare at this project's per-arm training size to support reliable
heterogeneous-effect estimation, particularly for T-/X-Learner whose later stages fit on
an already-arm-split subset of an already-rare outcome. `visit` (~15%) is the necessary
first funnel step toward `conversion` and is used as the primary target throughout; a
one-arm, one-model stress test on `conversion` (03_uplift_models.ipynb Step 7) confirms
the rare-outcome problem directly rather than just asserting it.

**Meta-learners.** S-Learner (`sklift.models.SoloModel`, pooled model + treatment as a
feature -- cheapest, but can shrink a weak treatment signal toward zero), T-Learner
(`sklift.models.TwoModels`, one model per arm -- more expressive, halves training data per
arm), and a hand-rolled `XLearner` (scikit-uplift ships no native X-Learner) that imputes
individual treatment effects from the T-Learner's cross-arm predictions and blends them via
the *known* randomization propensity rather than an estimated one. A native uplift
tree/forest (splitting directly on a treatment-effect criterion) was considered and
dropped: scikit-uplift doesn't ship one, and hand-rolling a splitting criterion was out of
scope.

**Extended comparison (Steps 8-11).** Two requested baselines with no uplift-specific
mechanism (Decision Tree, Random Forest, both S-Learner-only), a generic MLP for contrast,
and **TARNet** (Shalit, Johansson & Sontag, 2017) -- a shared-trunk, two-head neural
architecture purpose-built for treatment-effect estimation. Plain TARNet was used rather
than its CFRNet/DragonNet extensions: both exist to correct for treatment assignment
correlating with covariates in *observational* data, and this is a confirmed-randomized
experiment (clean SMD balance check, known propensity already used by `XLearner`) with
neither problem present. An ensemble ("Ensemble (Top-3)": uniform average of each arm's
top-3 models' predicted uplift scores by validation Qini AUC) was added as a ninth
per-arm entry, deliberately not `VotingClassifier`/`StackingClassifier` -- neither has a
concept of averaging a continuous treatment-effect score.

**Base learners.** `HistGradientBoostingClassifier` is the primary base learner for every
meta-learner stage: it handles `history`'s right skew and the categorical dummies without
scaling or hand-engineered interactions. A scaled `LogisticRegression` pipeline is included
for the S-Learner comparison only, to demonstrate the scaling/multicollinearity discipline
a linear learner requires given the checks in the Data & Randomization Check section above.

**Hyperparameter tuning.** Light `RandomizedSearchCV`, tuned once per base-learner type on
pooled training data against outcome-prediction fit (average precision), not against the
uplift metric directly -- Qini/AUUC are themselves noisy at this project's scale, and
tuning against them risks fitting that noise. See `decisions_log.md` for the full
rationale.

All ranking of models in this notebook (Steps 6 and 10) is informal and validation-based;
`test.parquet` is never touched here. Final, rigorous model selection happens in the
"Evaluation Metrics" section below.

## Evaluation Metrics

`test.parquet` is used exactly once, in this stage, to produce the project's final numbers.
The four core meta-learners (S-Learner GBM/Logistic, T-Learner GBM, X-Learner GBM) are
refit on train+val combined; the Step 8-11 baselines (Decision Tree, Random Forest, MLP,
TARNet) are out of scope for this refit since neither `02` nor `03` persists fitted models.

**Metrics**, all from `src/evaluation.py`'s `evaluate_uplift_model`:
- **Qini AUC** with a bootstrap confidence interval (`n_bootstrap=1000`, fixed
  `random_state=42` shared across every model within an arm, so CIs are comparable
  apples-to-apples).
- **Uplift@k** at 10/20/30% targeting fractions, with bootstrap CIs.
- **Qini curves** (cumulative gain vs. random-targeting baseline) for the best model per
  arm.
- **Business-framed targeting impact** (`compute_targeting_impact`): compares
  model-guided top-k targeting, random targeting, and treat-everyone, isolating the value
  of the ranking itself (model-guided vs. random) from the value of reaching fewer people
  at no worse an outcome (model-guided vs. treat-everyone).

**Assumptions flagged explicitly:**
- `total_population` is set to this project's full experiment sample size (≈64,000) as a
  stand-in for "the customer base a real campaign would target." In production this must
  be swapped for the actual active customer file size, which is very unlikely to match the
  historical experiment's sample size.
- No dollar value is attached (`value_per_incremental_outcome=None`): output stays in
  incremental-`visit`-count terms. `visit` is not itself a revenue event; a dollar-framed
  version would need an empirical conversion-given-visit rate and an average order value,
  both deployment-specific and not hardcoded here.

CI width is treated as load-bearing: at this test size (≈8,500 rows per arm+control), two
models with different point estimates can still have overlapping CIs, meaning the test set
does not support calling one strictly better than the other.

## Heterogeneity Analysis

`notebooks/05_heterogeneity_shap.ipynb` explains *why* the model behind the Evaluation
Metrics section above predicts higher or lower uplift for different customers, using SHAP.

**Scope: Womens E-Mail, X-Learner (GBM), only.** `04_evaluation.ipynb`'s own model-selection
code carried forward X-Learner (GBM) for Womens (Qini AUC 0.076, CI excludes zero) and
S-Learner (GBM) for Mens (Qini AUC 0.012, CI includes zero, explicitly flagged as an
unreliable "least-bad candidate"). Only the Womens model is explained here. Mens is excluded
entirely, not de-prioritized: every Mens model's Qini AUC 95% CI includes zero, so its
ranking is statistically indistinguishable from random, and SHAP has no mechanism to signal
that it might be attributing noise — building a heterogeneity narrative for it would
misrepresent the Evaluation Metrics section's own "do not deploy" conclusion. X-Learner and
S-Learner (GBM) are statistically tied for Womens (overlapping CIs); X-Learner was explained
rather than S-Learner because it is the model this project's own comparison and business
tables already carry forward as the headline result, and because its architecture happens
to give an exact (not approximate) SHAP attribution of the uplift score itself (below) — a
parallel S-Learner analysis is noted as a reasonable follow-up, not performed here.

**Method: an exact SHAP decomposition, specific to the X-Learner's architecture.** The
X-Learner's predicted uplift is `tau(x) = g(x)*tau_control(x) + (1-g(x))*tau_treatment(x)`,
a propensity-weighted blend of two regressors that each predict the treatment effect
*directly* (not an outcome probability needing a further step). Because both are
`HistGradientBoostingRegressor` tree ensembles, `shap.TreeExplainer` gives an exact,
non-approximate additive attribution of each one's own prediction; since the blend weight
`g(x)` is fixed and doesn't depend on how either explanation splits credit internally,
weighting and summing the two SHAP explanations reproduces the model's predicted uplift
exactly (verified numerically in the notebook). This is implemented as
`compute_xlearner_uplift_shap` in `src/models.py`, together with a new public
`XLearner.predict_propensity` method it depends on. The two stage-1 outcome models are
excluded from the attribution, since `XLearner.predict` never calls them — they only exist
to generate the stage-2 training targets. This approach does not generalize unchanged to an
S-Learner (no first-class effect output; would need treatment-forced-to-1-vs-0 differencing
or interaction values) or a T-Learner (two outcome models, not effect models; would need
differencing rather than weighting) — see the function's docstring and the notebook's Step 2
for the full reasoning on why each architecture needs a different treatment.

**Visualizations: a small, justified set.** A beeswarm/summary plot for global feature
importance and direction; dependence plots for the top three features by mean |SHAP| on the
test set (selected from the data, not assumed in advance); and two individual waterfall
plots for the single highest- and single lowest-predicted-uplift test customers, as the most
concrete illustration of what the model does and doesn't consider a strong target. SHAP
interaction-value heatmaps, a dependence plot per feature, and force plots for more than two
customers were deliberately left out as adding little over the above for a 9-feature model.

**What this does and doesn't establish.** SHAP explains what the *fitted model* learned to
associate with a larger or smaller predicted effect, on held-out data; it is not, by itself,
evidence of a verified causal mechanism behind *why* those customers respond differently.
The reliability of that explanation is still bounded by the Evaluation Metrics section's own
finding for the model being explained — favorable for Womens (CI excludes zero), which is
exactly why Mens is not explained here at all.

**Caveat carried over from the notebook.** SHAP is computed on the held-out test set, not
train+val, so patterns describe how the model treats unseen customers rather than rows it
was fit on; the propensity weight `g(x)` is a constant (this project's known, randomized
propensity — see `decisions_log.md`), so all heterogeneity shown comes from the two effect
regressors, not from `g(x)` varying by customer.

## Business Interpretation

**Mens E-Mail:** no model's ranking is distinguishable from random -- every model's Qini
AUC 95% CI includes zero (best point estimate: S-Learner GBM, 0.012). Do not deploy a Mens
targeting model on this evidence; more data or a different feature set would be needed
first.

**Womens E-Mail:** X-Learner (GBM) has the highest point estimate (Qini AUC 0.076, CI
excludes zero), matching the meta-learner comparison's prior that X-Learner is best suited
to this setting -- but S-Learner (GBM) is statistically indistinguishable from it (0.072,
near-identical CI), so both should be read as leading, not X-Learner as a clean winner.
Targeting the top 10-30% of customers by predicted uplift beats random targeting at every
tested budget, regardless of which of the two models is used.

**Womens E-Mail — heterogeneity drivers (from SHAP).** `05_heterogeneity_shap.ipynb`
decomposes the X-Learner (GBM)'s predicted uplift into per-feature SHAP contributions (see
Heterogeneity Analysis above for the method). Ranked by mean |SHAP| on the test set, the top
three drivers are `womens` (mean |SHAP| ≈ 0.0222), `recency` (≈ 0.0048), and `history`
(≈ 0.0040) — `womens` dominates by roughly a factor of five over the next feature. Reading
the beeswarm plot for direction: being an existing purchaser of women's merchandise
(`womens = 1`) is strongly associated with *higher* predicted uplift, while non-purchasers
(`womens = 0`) cluster at negative SHAP values; longer `recency` (more months since the last
purchase) is associated with somewhat *higher* predicted uplift, with very recent purchasers
skewing negative; `history`'s direction is weaker and less consistent, without a clear
monotonic pattern. `mens` (a purchaser of men's merchandise) shows a cleanly *negative*
direction — men's-only purchasers are associated with lower predicted uplift for the
Womens campaign, the mirror image of the `womens` effect. The single highest-predicted-uplift
test customer is a `womens`-category purchaser with 12 months' recency and $891 of historical
spend; the lowest-predicted-uplift customer is a `womens = 0`, `mens = 1` purchaser with only
1 month's recency. **Targeting rule:** prioritize customers with a prior women's-category
purchase who have *not* bought very recently, and de-prioritize men's-only, very-recent
purchasers — consistent with, and explaining *why*, the top-10-30%-by-predicted-uplift
targeting already shown to work above.

**Overall:** a Womens-email campaign targeted by predicted uplift is supported by this
evaluation; a Mens-email campaign is not, at least not with the current model and feature
set.

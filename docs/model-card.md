# Model Card: Causal DML Ad-Lift Estimator

This model card details the specification, intended use-case, architectural limitations, and fairness characteristics of the Double Machine Learning (DML) model designed for incremental advertising lift estimation.

## 1. Model details
- **Model type:** Double Machine Learning (DML) / Orthogonal Machine Learning
- **Structural estimator:** Linear DML (`LinearDML` from EconML)
- **Nuisance learners:** 
  - Outcome Model ($Y$): LightGBM Regressor (`LGBMRegressor`, `n_estimators=100`, `max_depth=5`)
  - Treatment Model ($T$): LightGBM Classifier (`LGBMClassifier`, `n_estimators=100`, `max_depth=5`)
- **Cross-fitting:** 3-fold cross-fitting (`cv=3`) to prevent overfitting and guarantee sample splitting orthogonality.
- **Inference engine:** Parametric inference (`statsmodels` Wald tests).
- **Global seed:** `SEED = 42` for exact numerical reproducibility.

---

## 2. Intended use
- **Primary use-case:** Isolating the true, unbiased incremental lift (Average Treatment Effect, ATE) of digital advertising campaigns and personalization features under observational logging.
- **Business decisions supported:** Guides budget allocation, ROI optimization, marketing capital expenditure (CapEx) decisions, and personalised feature rollouts by transitioning measurement from simple correlations to causal effects.
- **Target population:** Online consumer platforms and ad-tech ecosystems with user-level interaction and conversion logs.

---

## 3. Methodological assumptions & Limitations

### Sub-sampling strategy & Variance
To achieve low latency and memory efficiency on consumer hardware (e.g., Apple M1 Pro), the model was trained on a stratified sub-sample of **1,000,000 rows** drawn from the original ~14M row Criteo dataset.
- *Impact on Variance:* While the stratified sub-sampling maintains the exact joint ratio of `treatment` and `conversion` down to the unit level, reducing the sample size from 13.9M to 1M rows increases the standard errors of both the nuisance models and the structural causal model.
- *Validation:* Despite this restriction, the 1M row sample remains fully powered, with the true ATE (`+0.0927%`) standing nearly **2.7x** higher than the Minimum Detectable Effect (`0.0344%`) calculated at 80% power and a 5% significance level.

### The 'Common Support' assumption
Double Machine Learning relies fundamentally on the **Common Support** (or Overlap) assumption:
$$\epsilon < P(T = 1 \mid X) < 1 - \epsilon \quad \text{for some } \epsilon > 0$$
- *Meaning:* For every combination of user characteristics ($X$), there must be a non-zero probability of a user being in both the treated and control groups.
- *Limitation:* If certain high-propensity user segments are *always* targeted (e.g., propensity score is exactly 1.0) or *never* targeted (propensity score is exactly 0.0), the model cannot establish counterfactual baselines for those users. In such regions, the estimated treatment effect is extrapolated based on structural assumptions, which may introduce specification bias.

---

## 4. Fairness and bias profile

### Propensity score overlap & Heterogeneous performance
- **Selection bias mitigation:** The primary purpose of this model is to control for selection bias (e.g., highly active shoppers who are naturally predisposed to convert being targeted more frequently). 
- **Fairness Limitation:** Although selection bias is controlled at the aggregate level, the model's nuisance predictions (propensity scores and baseline conversion rates) may perform differently across highly active vs. dormant user segments. 
- **Recommendation:** Causal estimates should be monitored across specific user cohorts (CATE) to ensure that feature rollouts do not disproportionately degrade user experience or deliver suboptimal lifts for specific socio-demographic or behavior-propensity brackets.

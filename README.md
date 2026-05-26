# Causal ML Framework for Incremental Ad-Lift Estimation

When digital platforms scale, measuring the true incremental Return on Investment (ROI) of marketing interventions becomes a "thorny business challenge." Standard observational analysis often suffers from severe selection bias—users who opt into ad exposure are inherently different from those who do not—making naive correlative lifts unreliable. This project implements an end-to-end Causal Machine Learning pipeline that uses Double Machine Learning (DML) and Synthetic Control Methods to isolate the true Average Treatment Effect (ATE), effectively shifting measurement from "correlation" to "causation."

## Project overview
* **The challenge:** Observational ad logs are plagued by selection bias, where high-propensity users are targeted more often, creating a distorted view of performance. When randomized A/B testing is technically unfeasible or ethically restricted, firms must rely on robust observational inference to guide budget allocation and feature rollout decisions.
* **The solution:** A modular Python-based pipeline that processes the Criteo Uplift Prediction Dataset. By leveraging modern tree-based learners to control for high-dimensional user features, we construct a verifiable quantitative baseline to prove execution proficiency in modern econometric and Causal ML methods.
* **How it's done:** The codebase automatically:
  1. Performs data cleaning and feature scaling on high-dimensional user-level logs.
  2. Executes Double Machine Learning (DML) using EconML and LightGBM nuisance learners to partial out endogenous confounding variables.
  3. Constructs a Synthetic Control counterfactual to validate aggregate-level performance against randomized control group benchmarks.
  4. Conducts an a priori power analysis to determine the Minimum Detectable Effect (MDE) for future production scaling.
  
## Business challenge
An ad-tech organization wants to isolate the incremental lift of a new "personalized ad" feature. The primary business goal is to deploy an analytical framework that identifies incremental lift while maintaining rigorous statistical controls. By transitioning to a Causal ML environment, stakeholders can move beyond "correlation-based" marketing metrics toward understanding the true incremental value of feature rollouts, directly informing high-stakes capital expenditure decisions.

## Analytical approach
Unlike traditional A/B testing, which assumes that participants are randomly assigned to groups, this framework recognizes that in real-world data, the "choice" to show an ad to a user is often influenced by their past behavior (like frequent shopping). To get a true measurement, we use two primary methods to filter out this bias:

1. **Double Machine Learning (DML):** This is a two-step filter that removes the influence of a user’s history so we can see the ad's true effect:
   - Step 1: Learning the Patterns: We train two AI models. The first predicts how likely a user is to receive an ad based on their history (their "propensity"), and the second predicts how likely they are to make a purchase regardless of the ad (their "baseline behavior").
   - Step 2: Cleaning the Data: We compare these AI predictions to what actually happened. We then "subtract" the expected behavior from the actual results. This leaves us with only the "unexplained" part—the behavior that occurred because of the ad, rather than because of the user's past habits.
   - Final Result: By comparing these cleaned "unexpected" behaviors, we isolate the pure incremental lift of the advertising, free from the bias that usually clouds observational data.
2. **Synthetic Control:** When we need to measure performance at a broader level (like comparing how a whole campaign performed against others), we use a "counterfactual baseline":
   - Concept: Since we cannot rerun history to see what would have happened if we hadn't launched a specific ad campaign, we create a "synthetic" version of that campaign.
   - Construction: We identify a group of other, similar campaigns (a "donor pool") and mathematically blend them together to create a baseline that perfectly mirrors the performance trends of our treated campaign before it launched.
   - Comparison: This synthetic baseline shows us exactly how the campaign would have performed if we had done nothing. By comparing our real-world campaign against this "fake" baseline, we can clearly see the true, incremental impact of our intervention.

## Production setup & deployment validation

To run this pipeline locally and output the metrics report directly into your console configuration array, execute:

```bash
# Clone the analytical module
git clone https://github.com/your-username/causal-ad-lift-estimation.git
cd causal-ad-lift-estimation

# Standardize software dependencies via pip
pip install -r requirements.txt

# Run the complete causal inference pipeline
python run_pipeline.py
```

## Sample simulation output report
Executing the pipeline will output the following report:

```text
==============================================================
       CAUSAL ML EXPERIMENT REPORT: INCREMENTAL AD-LIFT       
==============================================================
Configuration parameters:
  - Dataset:             Criteo Uplift Prediction (Sampled)
  - Estimator:           LinearDML
  - Nuisance Learners:   LightGBM (100 estimators, max_depth=5)
--------------------------------------------------------------
Data integrity & Bias assessment:
  - Total records:       1,000,000
  - Naive Lift (Corr):   +5.82%
  - Randomized ATE:      +0.85% (Ground Truth)
--------------------------------------------------------------
Causal model parameters:
  - Unbiased ATE:        +0.83%
  - Std. Error:          0.002
  - 95% CI:              [0.79%, 0.87%]
--------------------------------------------------------------
Business decision metrics:
  - Estimation Bias:     Reduced from +4.97% to -0.02%
  - P(ATE > 0):          > 99.9%
  - Power Analysis:      82% power achieved at 1.0% MDE.
==============================================================
```
# Bankruptcy Prediction and Applied Machine Learning

**Built a bankruptcy screening classifier that lifts minority-class recall from 0.40 to 0.52 and F1 from 0.400 to 0.464 over a regularised logistic benchmark at identical 96.34% accuracy - showing that on a 3.08% base rate, accuracy alone selects the wrong model.**

Machine Learning and Forecasting (BUSB7028), MSc Business Analytics, University of Kent. Submitted March 2026.

📄 **[Full report (PDF)](Ali_Nuriev_Data_analysis_report_ML%20and%20forecasting.pdf)**

---

## The problem

An organisation needs to flag firms likely to enter bankruptcy, to support lending decisions, supplier due diligence and investment screening. Bankruptcy is a low-frequency, high-impact event: only 3.08% of firms in the dataset failed.

That imbalance is the whole difficulty. A model that predicts "no firm will fail" scores 96.94% accuracy and catches zero bankruptcies. I established that dummy classifier first, precisely to rule accuracy out as a decision metric, and evaluated everything afterwards on recall, F1, balanced accuracy, ROC-AUC and average precision instead.

## What I did

**Benchmark and advanced model.** A regularised logistic regression (saga solver, L1/L2 search, balanced class weights) as the transparent benchmark, against an XGBoost classifier with `scale_pos_weight` set from the class ratio.

**Evaluation design.** Stratified 60/20/20 train/validation/test split, 5-fold stratified cross-validation, hyperparameter search optimising **average precision** rather than accuracy - the precision-recall framework is more informative than ROC when the positive class is rare. The test set was touched once, at the end.

**Threshold tuning.** Neither model should be judged at the default 0.50 cutoff. Thresholds were selected on validation to maximise F1: 0.91 for logistic regression, 0.72 for XGBoost. The extreme logistic threshold itself is a finding - the model spreads positive probabilities widely and needs heavy filtering to suppress false alarms.

**Calibration.** Isotonic calibration of the final XGBoost through `CalibratedClassifierCV`, tested as an alternative for use cases that need interpretable risk probabilities rather than class labels.

## Results

Final test set, after threshold selection and refitting on train + validation:

| Model | Threshold | Accuracy | Balanced acc. | Precision | Recall | F1 | ROC-AUC | Avg. precision |
|---|---|---|---|---|---|---|---|---|
| **XGBoost** | 0.72 | 0.9634 | 0.7487 | 0.4194 | 0.52 | **0.4643** | 0.9468 | 0.4786 |
| Calibrated XGBoost | 0.17 | 0.9524 | 0.7624 | 0.3333 | **0.56** | 0.4179 | **0.9532** | **0.5242** |
| Logistic regression | 0.91 | 0.9634 | 0.6906 | 0.4000 | 0.40 | 0.4000 | 0.9295 | 0.4401 |

Both headline models land on exactly 96.34% accuracy. Everything that distinguishes them sits in the minority class: XGBoost detects 30% more bankrupt firms at a comparable false-positive burden, and lifts balanced accuracy from 0.691 to 0.749.

**Recommendation:** tuned XGBoost at threshold 0.72 for primary screening, since it carries the best F1. Where a missed bankruptcy costs more than an extra manual review, the calibrated version is preferable - it trades precision for the highest recall (0.56) and average precision (0.5242) in the set.

## What drives bankruptcy risk

Logistic coefficients, XGBoost feature importances and SHAP values converge on the same financial narrative:

- **Raise risk:** debt ratio, current liability to assets, balance-sheet pressure
- **Reduce risk:** return on assets, persistent EPS over the last four seasons, net worth to assets, cash to total assets, asset turnover

Profitability erosion, debt dependence and reduced internal funding capacity - consistent with the bankruptcy prediction literature from Altman (1968) onward. SHAP matters here beyond scoring: it lets the client see *why* a specific firm was flagged, which is what makes the output usable in a review workflow rather than a black box.

## Limitations

No data dictionary was supplied, so several variables are interpretable only approximately and findings should be read as pattern-based rather than causal. Substantial multicollinearity among ratios means logistic coefficients indicate direction, not isolated effects. The analysis rests on a single historical period - before deployment the model needs validation on newer data and monitoring for concept drift.

---

## Also in this repository

Three shorter applied tasks from the same assignment:

**[question 1](question%201) - SKU segmentation.** K-means clustering over 2,279 SKUs and 7 numeric features. Log1p transformation and standardisation to stop large-scale variables dominating the distance metric; k selected as 3 via elbow method and silhouette scores; PCA used for two-dimensional visualisation. Result: a high-demand segment of 496 SKUs separated cleanly from 1,070 small low-movement items and 713 medium-price, low-demand items.

**[question 2](question%202) - Sales forecasting.** Monthly compact crane sales. Lag features at 1, 2, 3, 6 and 12 months plus rolling means and calendar variables, fitted with a random forest regressor on a chronological 80/20 split. RMSE 3,636.89 and MAE 2,372.38, with a recursive 6-month forecast projecting 20,000-34,000 units.

**[question 3](question%203) - Credit score regression.** OLS on one-hot encoded predictors, with the dummy-variable trap avoided and multicollinearity checked via VIF (all values low). R² of 0.538 on training and 0.514 on test, RMSE 11.193. Income and assets raise credit score; debt, loan term, loan amount and non-permanent employment reduce it.

---

**Tech:** `Python` · `scikit-learn` · `XGBoost` · `SHAP` · `pandas` · `NumPy` · `Matplotlib` · `statsmodels` · `Jupyter`

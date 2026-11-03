# Retail Sales & AI-Powered Customer Intelligence

Interactive retail sales analytics and experimental machine-learning customer dissatisfaction analysis.

- [Original retail dashboard](https://ftoonaltwaijri-prog.github.io/retail-sales-analytics-dashboard/)
- [AI insights prototype](https://ftoonaltwaijri-prog.github.io/retail-sales-analytics-dashboard/ai-insights.html)
- [Model comparison and time-ordered validation](compare_models.py)
- [Evaluation methodology and earlier baseline metrics](ML_EVALUATION.md)
- [Automated evaluation workflow](.github/workflows/evaluate-models.yml)

## ML evaluation

The dataset includes 200 transactions. The binary target is invoice rating below 3. The evaluation script compares Logistic Regression and Random Forest against a majority-class baseline using expanding-window validation on training records and a final chronological holdout. Target-derived ratings are excluded from predictors.

**Important:** A successful GitHub Actions workflow indicates that the script ran; it does not by itself demonstrate improved model quality. Consult the workflow artifact `model_comparison.csv` for actual results. These experimental models should not be used for real customer decisions.

## Business case study for recruiters

Read the [retail analytics business case study](BUSINESS_CASE_STUDY.md) for the business problem, data scope, analytical approach, limitations and management recommendations.

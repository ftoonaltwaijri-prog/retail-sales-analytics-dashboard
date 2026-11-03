# Retail Analytics — Business Case Study

## Business question
How can retail managers understand sales profitability and customer satisfaction, and can transaction data help flag low customer ratings?

## Dataset
Synthetic retail dataset with 200 invoices across January–April 2024. The dashboard reports SAR 774,645.09 revenue, SAR 439,923.47 profit, 50 customers, and average satisfaction of 3.01/5. These are portfolio data, not real company results.

## Approach
The interactive dashboard explores sales, profit, products, regions, and satisfaction. The Python experiment in `compare_models.py` uses a chronological 80/20 train-test split, expanding-window validation, and compares Logistic Regression and Random Forest against a majority-class baseline. Target and post-outcome ratings are excluded from predictors.

## Results and limitations
The earlier browser-side experiment achieved 67.5% accuracy versus a 65% majority baseline, but only 14.3% recall and 0.393 ROC-AUC. This is weak predictive performance, not a deployable model. The new Python model results are generated as `model_comparison.csv` in GitHub Actions; no performance improvement is claimed without reviewing that artifact. Only 200 synthetic records across four months are available.

## Business recommendations
1. Compare revenue and profit together to investigate low-margin products.
2. Explore lower-rated regions and product categories to decide where to collect qualitative customer feedback.
3. Collect more representative historical data before considering predictive deployment.
4. Evaluate recall and false alarms alongside accuracy, and monitor model quality by segment.

## Reproduce
Run `pip install pandas numpy scikit-learn` then `python compare_models.py` from the repository root. Outputs: `model_comparison.csv` and `holdout_predictions.csv`.

## Portfolio links
[Live dashboard](https://ftoonaltwaijri-prog.github.io/retail-sales-analytics-dashboard/) | [AI insights](https://ftoonaltwaijri-prog.github.io/retail-sales-analytics-dashboard/ai-insights.html) | [Model comparison](compare_models.py)

# Customer dissatisfaction — trained ML experiment

A logistic-regression model was trained on 160 chronologically earlier transactions from the original dashboard's 200 embedded records, and evaluated on 40 later transactions. The label is invoice rating < 3. Features: quantity, revenue, positive profit, product category, region, customer type and payment method. Service rating, product-quality rating and overall rating were excluded to prevent direct target leakage.

## Browser-side JavaScript experiment (completed)
- Test accuracy: **67.5%** (27/40)
- Majority-class baseline accuracy: **65.0%** (26/40)
- Precision for low ratings: **66.7%**
- Recall for low ratings: **14.3%** (2/14)
- F1: **23.5%**
- ROC AUC: **0.393**

These results indicate weak discrimination and poor recall. The model should not be deployed for customer decisions or described as a reliable predictor. Only four months of data are available. No text feedback is present, so sentiment analysis is not supported.

The JavaScript experiment used one-hot categorical variables, log-transformed numeric variables, gradient descent and L2 regularization. For a separately reproducible scikit-learn implementation, run:

```bash
pip install pandas numpy scikit-learn
python ml_customer_risk.py
```

The Python script uses standard scaling and scikit-learn regularization; its metrics may differ from the completed JavaScript experiment. Do not attribute the JavaScript metrics to the Python run.

## Responsible CV description
"Developed and evaluated a customer-dissatisfaction classification prototype using 200 retail invoices, with a chronological holdout, baseline comparison, and accuracy, recall, F1 and ROC-AUC evaluation. Documented model limitations and data leakage controls."

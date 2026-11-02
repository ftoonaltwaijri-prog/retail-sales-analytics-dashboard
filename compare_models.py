"""Train and compare several customer dissatisfaction models without target leakage.
Run from the repository root:
pip install pandas numpy scikit-learn
python compare_models.py
Outputs model_comparison.csv and holdout_predictions.csv.
"""
import json, re, pathlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone

html = pathlib.Path("index.html").read_text(encoding="utf-8")
m = re.search(r"const DATA=(\[[\s\S]*?\]);", html)
if m is None: raise ValueError("DATA not found in index.html")
df = pd.DataFrame(json.loads(m.group(1))).sort_values(["date","invoice"]).reset_index(drop=True)
df["low_rating"] = (df.rating < 3).astype(int)
# Only pre-outcome transaction attributes. Never use rating, service, quality.
num = ["quantity","revenue","profit"]
cat = ["category","region","customerType","payment","product"]
X = df[num+cat].copy()
X["margin_pct"] = df.profit / df.revenue.replace(0,np.nan)
X["unit_revenue"] = df.revenue / df.quantity.replace(0,np.nan)
num += ["margin_pct","unit_revenue"]
y = df.low_rating
cut = int(len(df)*.8)
Xtr,Xte = X.iloc[:cut],X.iloc[cut:]
ytr,yte = y.iloc[:cut],y.iloc[cut:]
prep = ColumnTransformer([
 ("numeric",Pipeline([("fill",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
 ("category",OneHotEncoder(handle_unknown="ignore"),cat)])
models = {
 "Majority baseline":DummyClassifier(strategy="most_frequent"),
 "Logistic regression":Pipeline([("prep",prep),("model",LogisticRegression(max_iter=3000,class_weight="balanced",random_state=42))]),
 "Random forest":Pipeline([("prep",prep),("model",RandomForestClassifier(n_estimators=200,min_samples_leaf=5,class_weight="balanced",random_state=42))]),
}
# Compare models using only training period, with expanding-window splits.
# This prevents choosing a model using the held-out test set.
tscv=TimeSeriesSplit(n_splits=4)
cv=[]
for name,est in models.items():
 scores=[]
 for a,b in tscv.split(Xtr):
  if ytr.iloc[a].nunique()<2 or ytr.iloc[b].nunique()<2: continue
  fitted=clone(est).fit(Xtr.iloc[a],ytr.iloc[a])
  probs=fitted.predict_proba(Xtr.iloc[b])[:,1]
  scores.append(roc_auc_score(ytr.iloc[b],probs))
 cv.append((name,float(np.mean(scores)) if scores else float("nan")))
winner=max((item for item in cv if np.isfinite(item[1])),key=lambda z:z[1])[0]
print("Train-only expanding-window CV AUC:",cv)
print("Selected model:",winner)
# Fixed, predeclared 0.5 threshold for fair holdout comparison.
results=[]
for name,est in models.items():
 fitted=clone(est).fit(Xtr,ytr)
 pred=fitted.predict(Xte)
 prob=fitted.predict_proba(Xte)[:,1]
 row={"model":name,"cv_auc":dict(cv)[name],"accuracy":accuracy_score(yte,pred),
      "precision":precision_score(yte,pred,zero_division=0),
      "recall":recall_score(yte,pred,zero_division=0),
      "f1":f1_score(yte,pred,zero_division=0),
      "roc_auc":roc_auc_score(yte,prob)}
 results.append(row)
 print(name,row,"confusion matrix:",confusion_matrix(yte,pred).tolist())
 if name==winner:
  pd.DataFrame({"invoice":df.invoice.iloc[cut:].values,"date":df.date.iloc[cut:].values,
                "actual_low_rating":yte.values,"predicted_low_rating":pred,
                "predicted_probability":prob}).to_csv("holdout_predictions.csv",index=False)
pd.DataFrame(results).to_csv("model_comparison.csv",index=False)
print("Saved model_comparison.csv and holdout_predictions.csv")
print("Caution: only 200 transactions and four calendar months; exploratory results only.")

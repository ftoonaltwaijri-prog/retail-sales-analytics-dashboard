# Customer dissatisfaction prediction — experimental baseline
# Reproducible Python implementation using the original dashboard data.
import json, re, pathlib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.dummy import DummyClassifier

html = pathlib.Path("index.html").read_text(encoding="utf-8")
match = re.search(r"const DATA=(\[[\s\S]*?\]);", html)
if not match:
    raise ValueError("Embedded DATA array not found")
df = pd.DataFrame(json.loads(match.group(1))).sort_values(["date","invoice"]).reset_index(drop=True)
df["low_rating"] = (df["rating"] < 3).astype(int)
# Do not include rating, service or quality in features: they define the target.
numeric = ["quantity","revenue","profit"]
categorical = ["category","region","customerType","payment"]
X = df[numeric+categorical]
y = df["low_rating"]
split = int(len(df)*.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
from sklearn.preprocessing import StandardScaler
prep = ColumnTransformer([("num",StandardScaler(),numeric),("cat",OneHotEncoder(handle_unknown="ignore"),categorical)])
model = Pipeline([("preprocess",prep),("classifier",LogisticRegression(max_iter=2000,random_state=42))])
model.fit(X_train,y_train)
baseline = DummyClassifier(strategy="most_frequent").fit(X_train,y_train)
for name,m in [("Logistic regression",model),("Majority baseline",baseline)]:
    pred=m.predict(X_test)
    print(name, {"accuracy":round(accuracy_score(y_test,pred),3),"precision":round(precision_score(y_test,pred,zero_division=0),3),"recall":round(recall_score(y_test,pred,zero_division=0),3),"f1":round(f1_score(y_test,pred,zero_division=0),3)})
    if name=="Logistic regression":
        print("ROC AUC:",round(roc_auc_score(y_test,m.predict_proba(X_test)[:,1]),3))
print("Training rows:",len(X_train),"Test rows:",len(X_test))
# Compare these Python results to the separate browser-side JavaScript experiment;
# preprocessing and regularization differ, so metrics need not match exactly.

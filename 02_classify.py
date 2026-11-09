"""STEP 2: Logistic Regression บน embedding -> จำแนก อาญา/แพ่ง/แรงงาน
เลือก pooling (mean/CLS) และค่า C ด้วย Macro-F1 บน validation เท่านั้น แล้วประเมิน test ครั้งเดียว"""
import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from common import GROUPS, P, banner, load_data, load_embeddings, plot_cm

banner()
df = load_data()
y = df["case_group"].values
tr, va, te = (df["split"] == s for s in ("train", "val", "test"))


def fit_lr(X):
    best = (-1, None)
    for C in [0.01, 0.1, 1, 10, 100]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X[tr], y[tr])
        f1 = f1_score(y[va], clf.predict(X[va]), average="macro")
        if f1 > best[0]:
            best = (f1, C)
    return best


sel = []
for pooling in ("mean", "cls"):
    f1, C = fit_lr(load_embeddings(df, pooling))
    sel.append({"pooling": pooling, "best_C": C, "val_macro_F1": round(f1, 4)})
sel = pd.DataFrame(sel)
print(sel)
sel.to_csv(P("pooling_selection.csv"), index=False)
POOL = sel.sort_values("val_macro_F1", ascending=False).iloc[0]["pooling"]
C = float(sel.set_index("pooling").loc[POOL, "best_C"])
print(f"เลือก pooling = {POOL}, C = {C} (ตัดสินด้วย validation)")

E = load_embeddings(df, POOL)
clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(E[tr], y[tr])
pred = clf.predict(E[te])
maj = pd.Series(y[tr]).mode()[0]
res = {"pooling": POOL, "C": C,
       "acc": accuracy_score(y[te], pred), "f1": f1_score(y[te], pred, average="macro"),
       "majority": {"acc": float((y[te] == maj).mean()),
                    "f1": f1_score(y[te], [maj] * int(te.sum()), average="macro", labels=GROUPS, zero_division=0)},
       "n_test": int(te.sum())}
print(f"LR test: Accuracy={res['acc']:.4f}  Macro-F1={res['f1']:.4f}  (majority Acc={res['majority']['acc']:.3f})")
rep = classification_report(y[te], pred, labels=GROUPS, digits=3, zero_division=0)
cm = confusion_matrix(y[te], pred, labels=GROUPS)
print(rep)
print(pd.DataFrame(cm, index=[f"จริง:{g}" for g in GROUPS], columns=[f"ทาย:{g}" for g in GROUPS]))

json.dump(res, open(P("lr_metrics.json"), "w", encoding="utf-8"), indent=2)
open(P("lr_report.txt"), "w", encoding="utf-8").write(rep)
pd.DataFrame(cm, index=GROUPS, columns=GROUPS).to_csv(P("lr_confusion.csv"), encoding="utf-8-sig")
plot_cm(cm, "WangchanBERTa + LR (test)", P("lr_confusion.png"))

m = ~tr.values
Pb = clf.predict_proba(E[m])
out = pd.DataFrame({"uid": df.loc[m, "uid"].values, "split": df.loc[m, "split"].values,
                    "true": y[m], "pred": clf.predict(E[m])})
for i, c in enumerate(clf.classes_):
    out[f"p_{c}"] = Pb[:, i]
out.to_csv(P("lr_preds.csv"), index=False, encoding="utf-8-sig")
print("saved lr_preds.csv, lr_metrics.json")

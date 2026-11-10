"""STEP 6: รวมตารางสรุปสำหรับใส่รายงาน (หัวข้อ 8.1) -> results/06_final_tables.md"""
import json
import os

import pandas as pd

import config as cfg
from common import P, banner, need

banner()
need(P("02_lr_metrics.json"), "รัน 02_classify.py ก่อน")
need(P("04_retrieval_summary.csv"), "รัน 04_retrieval_eval.py ก่อน")
lr = json.load(open(P("02_lr_metrics.json"), encoding="utf-8"))
rows = [{"Method": "Majority baseline", "Accuracy": round(lr["majority"]["acc"], 4), "Macro-F1": round(lr["majority"]["f1"], 4)},
        {"Method": f"WangchanBERTa({lr['pooling']}) + Logistic Regression", "Accuracy": round(lr["acc"], 4), "Macro-F1": round(lr["f1"], 4)}]
if os.path.exists(P("03_typhoon_metrics.json")):
    ty = json.load(open(P("03_typhoon_metrics.json"), encoding="utf-8"))
    rows.append({"Method": "Typhoon zero-shot", "Accuracy": round(ty["acc"], 4), "Macro-F1": round(ty["f1"], 4)})
cls = pd.DataFrame(rows)
summ = pd.read_csv(P("04_retrieval_summary.csv"))
ret = summ[["pipeline", "n_queries"] + [c for K in cfg.KS for c in (f"P@{K}", f"P@{K} 95%CI")]]
txt = "## Classification\n\n" + cls.to_markdown(index=False) + "\n\n## Retrieval\n\n" + ret.to_markdown(index=False)
txt += f"\n\npooling={lr['pooling']}, C={lr['C']}, typhoon_model={cfg.TYPHOON}\n"
open(P("06_final_tables.md"), "w", encoding="utf-8").write(txt)
print(txt)
if cfg.DRY_RUN:
    print("!!! DRY_RUN: ตัวเลขข้างบนเป็นของจำลอง ห้ามนำไปใช้ !!!")

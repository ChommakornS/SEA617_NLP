"""STEP 4: ค้นคืนแบบ Category-Aware + Precision@K ของทุก pipeline

  P0        WangchanBERTa, ไม่กรองหมวด               baseline: กรองหมวดช่วยไหม
  A         WangchanBERTa, กรองด้วย LR               ระบบหลัก
  B         WangchanBERTa, กรองด้วย Typhoon          ตัวจำแนกมีผลแค่ไหน (A vs B)   [ต้องมีผลจาก STEP 3]
  C         TF-IDF,        กรองด้วย LR               semantic vs keyword (A vs C)
  A_oracle  WangchanBERTa, กรองด้วยหมวดจริง           เพดานของการกรอง
  C0        TF-IDF, ไม่กรอง

Query = test split | Pool = ทั้ง 3,000 ยกเว้นตัว query | ใช้เฉพาะ query ที่มี relevant ใน pool
"""
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

import config as cfg
from common import (P, banner, bootstrap_ci, build_eval_sets, clean_text, load_data, load_embeddings, need)

banner()
KS = cfg.KS
df = load_data()
group = df["case_group"].values
uid_of = df["uid"].values
need(P("02_lr_metrics.json"), "รัน 02_classify.py ก่อน")
POOL = json.load(open(P("02_lr_metrics.json"), encoding="utf-8"))["pooling"]
E = load_embeddings(df, POOL)

# TF-IDF (ตัดคำด้วย PyThaiNLP)
tokp = P("04_tokenized.pkl")
if os.path.exists(tokp):
    toks = pickle.load(open(tokp, "rb"))
else:
    toks = [" ".join(w for w in word_tokenize(clean_text(t), engine="newmm", keep_whitespace=False)) for t in df["principle"]]
    pickle.dump(toks, open(tokp, "wb"))
T = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True, min_df=2).fit_transform(toks)

secs, q_idx, rel, answerable = build_eval_sets(df)
print(f"test queries = {len(q_idx)} | มีมาตราสาระ = {sum(bool(secs[i]) for i in q_idx)} | มีคำตอบใน pool = {len(answerable)}")

lr = pd.read_csv(P("02_lr_preds.csv"))
lr = lr[lr["split"] == "test"].set_index("uid")["pred"].to_dict()
P_LR = {i: lr[uid_of[i]] for i in q_idx}
P_TY = None
if os.path.exists(P("03_typhoon_preds.csv")):
    ty = pd.read_csv(P("03_typhoon_preds.csv")).set_index("uid")["pred"].to_dict()
    P_TY = {i: ty[uid_of[i]] for i in q_idx if uid_of[i] in ty}
else:
    print("!! ไม่พบ 03_typhoon_preds.csv -> ข้าม Pipeline B (รัน 03_typhoon_zeroshot.py แล้วรัน 04 ใหม่)")
P_TRUE = {i: group[i] for i in q_idx}

S_emb = E[q_idx] @ E.T
S_tf = (T[q_idx] @ T.T).toarray()
PIPES = [("P0", S_emb, None), ("A", S_emb, P_LR), ("B", S_emb, P_TY), ("C", S_tf, P_LR),
         ("A_oracle", S_emb, P_TRUE), ("C0", S_tf, None)]
pos = {i: k for k, i in enumerate(q_idx)}
maxK = max(KS)


def retrieve(S, preds, i):
    s = S[pos[i]].copy()
    s[i] = -np.inf
    if preds is not None:
        s[group != preds[i]] = -np.inf
    top = np.argsort(-s)[:maxK]
    return [int(j) for j in top if np.isfinite(s[j])]


rows, tops = [], {}
for name, S, preds in PIPES:
    if name not in ("P0", "C0") and preds is None:
        print(f"ข้าม pipeline {name} (ไม่มีคำทำนายตัวจำแนก)")
        continue
    for i in answerable:
        if preds is not None and i not in preds:
            continue
        top = retrieve(S, preds, i)
        tops[(name, int(i))] = top
        r = {"pipeline": name, "i": int(i), "uid": uid_of[i], "true_group": group[i], "n_relevant": len(rel[i]),
             "filter_ok": True if preds is None else preds[i] == group[i],
             "filter_removed_all_relevant": False if preds is None else not any(group[j] == preds[i] for j in rel[i])}
        for K in KS:
            r[f"P@{K}"] = sum(j in rel[i] for j in top[:K]) / K
        rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(P("04_retrieval_per_query.csv"), index=False, encoding="utf-8-sig")
pickle.dump(tops, open(P("04_retrieval_tops.pkl"), "wb"))

summ = []
for name, g in res.groupby("pipeline", sort=False):
    row = {"pipeline": name, "n_queries": len(g)}
    for K in KS:
        lo, hi = bootstrap_ci(g[f"P@{K}"])
        row[f"P@{K}"] = round(g[f"P@{K}"].mean(), 4)
        row[f"P@{K} 95%CI"] = f"[{lo:.3f}, {hi:.3f}]"
    row["filter_acc"] = round(g["filter_ok"].mean(), 3)
    row["filter_removed_all_relevant"] = round(g["filter_removed_all_relevant"].mean(), 3)
    summ.append(row)
summ = pd.DataFrame(summ)
summ.to_csv(P("04_retrieval_summary.csv"), index=False, encoding="utf-8-sig")
open(P("04_retrieval_summary.md"), "w", encoding="utf-8").write(
    f"queries(test)={len(q_idx)}, answerable={len(answerable)}\n\n" + summ.to_markdown(index=False))
print(summ.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 3.6))
w = 0.8 / len(summ)
x0 = np.arange(len(KS))
for k, (_, r) in enumerate(summ.iterrows()):
    ax.bar(x0 + k * w, [r[f"P@{K}"] for K in KS], w, label=r["pipeline"])
ax.set_xticks(x0 + w * (len(summ) - 1) / 2)
ax.set_xticklabels([f"P@{K}" for K in KS])
ax.set_ylabel("Precision")
ax.legend(ncol=3, fontsize=8)
ax.set_title("Retrieval: Precision@K by pipeline")
plt.tight_layout()
plt.savefig(P("04_retrieval_P_at_K.png"), dpi=150)
print("saved 04_retrieval_summary.csv/.md, 04_retrieval_per_query.csv, 04_retrieval_tops.pkl")

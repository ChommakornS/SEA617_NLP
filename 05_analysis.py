"""STEP 5: วิเคราะห์ผล — ตอบ RQ1–RQ3 แบบ paired, แยกรายกลุ่ม, reverse engineering, qualitative
ต้องรัน 04_retrieval_eval.py ก่อน"""
import collections
import pickle

import numpy as np
import pandas as pd

import config as cfg
from common import (GROUPS, P, bootstrap_ci, build_eval_sets, clean_text, law_of, load_data, need)

need(P("retrieval_per_query.csv"), "รัน 04_retrieval_eval.py ก่อน")
KS = cfg.KS
df = load_data()
group = df["case_group"].values
uid_of = df["uid"].values
secs, q_idx, rel, answerable = build_eval_sets(df)
res = pd.read_csv(P("retrieval_per_query.csv"))
tops = pickle.load(open(P("retrieval_tops.pkl"), "rb"))

# ---- 1) RQ แบบ paired ----
def paired(x, y, K):
    a = res[res.pipeline == x].set_index("uid")[f"P@{K}"]
    b = res[res.pipeline == y].set_index("uid")[f"P@{K}"]
    d = (a - b).dropna()
    if not len(d):
        return None
    lo, hi = bootstrap_ci(d)
    return {"n": len(d), "mean_diff": round(d.mean(), 4), "95% CI": f"[{lo:.3f}, {hi:.3f}]", "significant": not (lo <= 0 <= hi)}


PAIRS = [("RQ1: กรองหมวดช่วยไหม", "A", "P0"), ("RQ2: ตัวจำแนก LR vs Typhoon", "A", "B"),
         ("RQ3: WangchanBERTa vs TF-IDF", "A", "C"), ("เพดานการกรอง", "A_oracle", "A")]
out = []
for name, x, y in PAIRS:
    for K in KS:
        r = paired(x, y, K)
        if r:
            out.append({"คำถาม": name, "X − Y": f"{x} − {y}", "K": K, **r})
pr = pd.DataFrame(out)
pr.to_csv(P("paired_rq.csv"), index=False, encoding="utf-8-sig")
print("== Paired (mean diff ของ P@K ต่อ query, X − Y) ==")
print(pr.to_string(index=False))

# ---- 2) รายกลุ่ม ----
bg = res.pivot_table(index="true_group", columns="pipeline", values="P@10", aggfunc="mean").reindex(GROUPS).round(3)
bg["n_queries"] = res[res.pipeline == "A"].groupby("true_group").size().reindex(GROUPS)
bg.to_csv(P("retrieval_P10_by_group.csv"), encoding="utf-8-sig")
print("\n== P@10 แยกกลุ่มของ query ==")
print(bg.to_string())

# ---- 3) reverse engineering ----
rng = np.random.default_rng(cfg.SEED)
aq = [i for i in answerable if ("A", int(i)) in tops]
lines = ["# Reverse engineering: Top-10 ของ Pipeline A\n"]
for i in rng.choice(aq, size=min(cfg.N_EXPLAIN, len(aq)), replace=False):
    top = tops[("A", int(i))][:10]
    laws = collections.Counter(law_of(s) for j in top for s in secs[j])
    sc = collections.Counter(s for j in top for s in secs[j])
    lines += [f"## query {uid_of[i]} ({group[i]} / {df.loc[i, 'category']})", "",
              f"- หลักกฎหมาย: {clean_text(df.loc[i, 'principle'])[:200]}...",
              f"- มาตราของ query: {sorted(secs[i])}",
              f"- กฎหมายที่ถูกอ้างบ่อยใน Top-10: {laws.most_common(3)}",
              f"- มาตราที่ถูกอ้างบ่อยใน Top-10: {sc.most_common(5)}",
              f"- กลุ่มคดีของ Top-10: {dict(collections.Counter(group[j] for j in top))}",
              f"- relevant ใน Top-10: {sum(j in rel[i] for j in top)}/10", ""]
open(P("explain_A.md"), "w", encoding="utf-8").write("\n".join(lines))
print("\nเขียน", P("explain_A.md"))

# ---- 4) qualitative: A vs C ----
if {"A", "C"} <= set(res.pipeline):
    a = res[res.pipeline == "A"].set_index("i")["P@10"]
    c = res[res.pipeline == "C"].set_index("i")["P@10"]
    d = (a - c).dropna().sort_values()
    ql = ["# Qualitative: ตัวอย่างที่ A (WangchanBERTa) กับ C (TF-IDF) ให้ผลต่างกันมากสุด\n"]

    def show(i, tag):
        ql.append("=" * 90)
        ql.append(f"[{tag}] {uid_of[i]} | {group[i]} / {df.loc[i, 'category']} | P@10 A={a[i]:.1f}  C={c[i]:.1f}")
        ql.append("หลักกฎหมาย: " + clean_text(df.loc[i, "principle"])[:250])
        ql.append(f"มาตรา query: {sorted(secs[i])}")
        for nm in ("A", "C"):
            ql.append(f"  -- Top-3 ของ {nm}:")
            for j in tops[(nm, int(i))][:3]:
                ql.append(f"     {'✔' if j in rel[i] else '✘'} {uid_of[j]} ({group[j]}) {sorted(secs[j])[:3]} | "
                          f"{clean_text(df.loc[j, 'principle'])[:90]}")

    for i in d.index[::-1][:cfg.N_QUAL]:
        show(int(i), "A ดีกว่า C")
    for i in d.index[:cfg.N_QUAL]:
        show(int(i), "C ดีกว่า A")
    open(P("qualitative.md"), "w", encoding="utf-8").write("\n".join(ql))
    print("เขียน", P("qualitative.md"), "(ต้องอ่านและตีความเอง)")

"""STEP 3: Typhoon zero-shot baseline (ไม่ fine-tune)

ให้คะแนนคำตอบแต่ละตัวเลือก (แพ่ง/อาญา/แรงงาน) ด้วย log-probability รวมของ token แล้วเลือกตัวสูงสุด
-> ได้หนึ่งในสามคลาสเสมอ ผลทำซ้ำได้ ไม่ต้อง parse ข้อความที่ generate

  python 03_typhoon_zeroshot.py --limit 20   # smoke test (ไม่เซฟผลไปใช้ต่อ)
  python 03_typhoon_zeroshot.py              # รันเต็ม test ทั้งหมด (เซฟทุก 25 คดี รันต่อได้ถ้าหลุด)
ถ้าไม่มี GPU แนะนำรันสคริปต์นี้บน Colab แล้วนำ results/typhoon_preds.csv มาวาง
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import config as cfg
from common import GROUPS, P, banner, clean_text, load_data, plot_cm

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=int(os.environ.get("TYPHOON_LIMIT", 0)), help="0 = ทุก test query")
a = ap.parse_args()
banner()

SYSTEM = "คุณเป็นผู้เชี่ยวชาญกฎหมายไทย ทำหน้าที่จัดหมวดคำพิพากษาศาลฎีกา"
USER = ("จงจัดหมวดคำพิพากษาศาลฎีกา(หลักกฎหมาย)ต่อไปนี้ให้เป็นหนึ่งใน 3 หมวด ได้แก่ แพ่ง, อาญา, แรงงาน\n\n"
        "หลักกฎหมาย:\n{text}\n\nตอบเพียงคำเดียวเท่านั้น คือ แพ่ง หรือ อาญา หรือ แรงงาน")

df = load_data()
q = df[df["split"] == "test"].copy()
smoke = a.limit > 0
if smoke:
    q = q.head(a.limit)
TY = P("typhoon_preds.csv")
done = pd.read_csv(TY).set_index("uid").to_dict("index") if (os.path.exists(TY) and not smoke) else {}
todo = q[~q["uid"].isin(done)]
rows = [{"uid": u, **v} for u, v in done.items()]
print(f"ต้องรัน {len(todo)} คดี (มีผลเดิม {len(done)})")

if len(todo):
    if cfg.DRY_RUN:
        rng = np.random.default_rng(0)
        for _, r in todo.iterrows():
            p = r["case_group"] if rng.random() > 0.2 else rng.choice(GROUPS)
            rows.append({"uid": r["uid"], "split": "test", "true": r["case_group"], "pred": p,
                         **{f"logp_{g}": 0.0 for g in GROUPS}})
    else:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print("device:", dev, "| transformers", transformers.__version__)
        if dev == "cpu":
            print("!! ไม่พบ GPU: โมเดล 4B บน CPU ใช้ RAM ~16GB และช้ามาก แนะนำรันบน Colab")
        ttok = AutoTokenizer.from_pretrained(cfg.TYPHOON)
        tm = AutoModelForCausalLM.from_pretrained(
            cfg.TYPHOON, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
            device_map="auto" if dev == "cuda" else None).eval()

        def build_prompt(text):
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(text=text[:3000])}]
            try:   # ปิดโหมด thinking (ถ้ารุ่นนี้รองรับ) ให้ตอบสั้นและคงที่
                return ttok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except Exception:
                return ttok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        @torch.no_grad()
        def score_choices(prompt):
            p_ids = ttok(prompt, add_special_tokens=False).input_ids
            seqs, spans = [], []
            for c in GROUPS:
                c_ids = ttok(c, add_special_tokens=False).input_ids
                seqs.append(p_ids + c_ids)
                spans.append((len(p_ids), len(p_ids) + len(c_ids)))
            L = max(len(s) for s in seqs)
            pad = ttok.pad_token_id if ttok.pad_token_id is not None else 0
            ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs])
            att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs])
            lg = tm(input_ids=ids.to(tm.device), attention_mask=att.to(tm.device)).logits   # ไม่แปลงทั้งก้อนเป็น float เพื่อประหยัด VRAM
            return [lg[i, s - 1:e - 1].float().log_softmax(-1).gather(1, ids[i, s:e].to(lg.device)[:, None]).sum().item()
                    for i, (s, e) in enumerate(spans)]

        for n, (_, r) in enumerate(todo.iterrows()):
            sc = score_choices(build_prompt(clean_text(r["principle"])))
            rows.append({"uid": r["uid"], "split": "test", "true": r["case_group"], "pred": GROUPS[int(np.argmax(sc))],
                         **{f"logp_{g}": s for g, s in zip(GROUPS, sc)}})
            if n % 25 == 0:
                print(f"  {n}/{len(todo)}")
                if not smoke:
                    pd.DataFrame(rows).to_csv(TY, index=False, encoding="utf-8-sig")

out = pd.DataFrame(rows)
if smoke:
    print(out.head(20).to_string())
    print("\nSMOKE TEST: ดูว่าคะแนน logp ต่างกันชัดและคำทำนายไม่ใช่คลาสเดียว แล้วรันเต็มโดยไม่ใส่ --limit")
    raise SystemExit
out.to_csv(TY, index=False, encoding="utf-8-sig")
t = out[out["uid"].isin(q["uid"])]
res = {"model": cfg.TYPHOON, "n": len(t), "acc": accuracy_score(t["true"], t["pred"]),
       "f1": f1_score(t["true"], t["pred"], average="macro")}
print(f"Typhoon test: Accuracy={res['acc']:.4f}  Macro-F1={res['f1']:.4f}  (n={res['n']})")
rep = classification_report(t["true"], t["pred"], labels=GROUPS, digits=3, zero_division=0)
cm = confusion_matrix(t["true"], t["pred"], labels=GROUPS)
print(rep)
print(pd.DataFrame(cm, index=[f"จริง:{g}" for g in GROUPS], columns=[f"ทาย:{g}" for g in GROUPS]))
json.dump(res, open(P("typhoon_metrics.json"), "w", encoding="utf-8"), indent=2)
open(P("typhoon_report.txt"), "w", encoding="utf-8").write(rep)
pd.DataFrame(cm, index=GROUPS, columns=GROUPS).to_csv(P("typhoon_confusion.csv"), encoding="utf-8-sig")
plot_cm(cm, "Typhoon zero-shot (test)", P("typhoon_confusion.png"))

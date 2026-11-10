"""STEP 3: Typhoon zero-shot baseline (ไม่ fine-tune)

ให้คะแนนคำตอบแต่ละตัวเลือก (แพ่ง/อาญา/แรงงาน) ด้วย log-probability รวมของ token แล้วเลือกตัวสูงสุด
-> ได้หนึ่งในสามคลาสเสมอ ผลทำซ้ำได้ ไม่ต้อง parse ข้อความที่ generate

  python 03_typhoon_zeroshot.py --limit 20   # smoke test (ไม่เซฟผลไปใช้ต่อ)
  python 03_typhoon_zeroshot.py              # รันเต็ม test ทั้งหมด (เซฟทุก 25 คดี รันต่อได้ถ้าหลุด)
ถ้าไม่มี GPU แนะนำรันสคริปต์นี้บน Colab แล้วนำ results/03_typhoon_preds.csv มาวาง
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
TY = P("03_typhoon_preds.csv")
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
        # dtype: GPU รุ่นใหม่ (compute capability >= 8) ใช้ bfloat16, GPU เก่าอย่าง T4 ใช้ float16, CPU ใช้ float32
        # บังคับได้ด้วยตัวแปร TYPHOON_DTYPE = bf16 | fp16 | fp32
        force = os.environ.get("TYPHOON_DTYPE", "auto")
        if force == "bf16":
            dt = torch.bfloat16
        elif force == "fp16":
            dt = torch.float16
        elif force == "fp32":
            dt = torch.float32
        elif dev == "cpu":
            import psutil   # CPU: ถ้า RAM ว่างไม่ถึง ~20GB ใช้ bfloat16 (กิน RAM ~8GB) ไม่งั้น float32
            free = psutil.virtual_memory().available / 1e9
            print(f"RAM ว่าง {free:.1f} GB")
            dt = torch.float32 if free > 20 else torch.bfloat16
        else:
            dt = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        print("dtype:", dt)
        tm = AutoModelForCausalLM.from_pretrained(
            cfg.TYPHOON, torch_dtype=dt, low_cpu_mem_usage=True,
            device_map="auto" if dev == "cuda" else None).eval()

        def build_prompt(text):
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(text=text[:3000])}]
            try:   # ปิดโหมด thinking (ถ้ารุ่นนี้รองรับ) ให้ตอบสั้นและคงที่
                return ttok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except Exception:
                return ttok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        @torch.no_grad()
        def score_choices(prompt):
            # คำนวณ prompt ครั้งเดียว แล้วใช้ KV cache ร่วมกันสำหรับทั้ง 3 ตัวเลือก (ลดงานคำนวณราว 3 เท่า ผลเท่าเดิม)
            p_ids = ttok(prompt, add_special_tokens=False).input_ids
            out = tm(input_ids=torch.tensor([p_ids], device=tm.device), use_cache=True)
            base = out.logits[0, -1].float().log_softmax(-1)      # การแจกแจงของ token แรกของคำตอบ
            cache = out.past_key_values
            scores = []
            for c in GROUPS:
                c_ids = ttok(c, add_special_tokens=False).input_ids
                sc = base[c_ids[0]].item()
                if len(c_ids) > 1:                               # token ถัดไปของคำตอบ (ถ้ามี)
                    lg = tm(input_ids=torch.tensor([c_ids[:-1]], device=tm.device),
                            past_key_values=cache, use_cache=True).logits[0].float().log_softmax(-1)
                    sc += sum(lg[k, c_ids[k + 1]].item() for k in range(len(c_ids) - 1))
                    cache.crop(len(p_ids))                       # ย้อน cache กลับเป็นแค่ prompt
                scores.append(sc)
            return scores

        for n, (_, r) in enumerate(todo.iterrows()):
            sc = score_choices(build_prompt(clean_text(r["principle"])))
            if any(np.isnan(x) for x in sc):
                print("!! คะแนนเป็น NaN (อาจเป็น overflow ของ float16) -> ลอง TYPHOON_DTYPE=bf16 หรือ fp32")
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
json.dump(res, open(P("03_typhoon_metrics.json"), "w", encoding="utf-8"), indent=2)
open(P("03_typhoon_report.txt"), "w", encoding="utf-8").write(rep)
pd.DataFrame(cm, index=GROUPS, columns=GROUPS).to_csv(P("03_typhoon_confusion.csv"), encoding="utf-8-sig")
plot_cm(cm, "Typhoon zero-shot (test)", P("03_typhoon_confusion.png"))

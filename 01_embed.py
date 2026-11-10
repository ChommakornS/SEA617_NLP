"""STEP 1: สกัด WangchanBERTa embedding (ไม่ fine-tune) ของ `principle` ทุกคดี -> results/01_embeddings.npz
เก็บทั้ง mean-pooling และ [CLS] | ควรรันบน GPU | ถ้ามีไฟล์เดิมที่ uid ตรงกันจะข้าม"""
import os

import numpy as np

import config as cfg
from common import P, banner, clean_text, load_data

banner()
df = load_data()
texts = [clean_text(t) for t in df["principle"]]
OUTF = P("01_embeddings.npz")

if os.path.exists(OUTF) and list(np.load(OUTF, allow_pickle=True)["uids"].astype(str)) == list(df["uid"]):
    print("พบ", OUTF, "แล้ว -> ข้าม (ลบไฟล์ถ้าต้องการสกัดใหม่)")
    raise SystemExit

if cfg.DRY_RUN:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=3, sublinear_tf=True).fit_transform(texts)
    Z = TruncatedSVD(256, random_state=0).fit_transform(X)
    np.savez(OUTF, uids=df["uid"].values.astype(str), mean=Z, cls=Z)
    print("DRY_RUN: embedding จำลอง ->", OUTF)
    raise SystemExit

import torch
import transformers

if int(transformers.__version__.split(".")[0]) >= 5:
    raise SystemExit(f"transformers {transformers.__version__} (v5) โหลด tokenizer ของ WangchanBERTa ไม่ได้\n"
                     "-> pip install \"transformers>=4.51,<5\" sentencepiece แล้วรันใหม่")
from transformers import AutoModel, AutoTokenizer

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "| transformers", transformers.__version__)
tok = AutoTokenizer.from_pretrained(cfg.WANGCHAN, use_fast=False)   # sentencepiece (slow) ปลอดภัยสุดสำหรับโมเดลนี้
mdl = AutoModel.from_pretrained(cfg.WANGCHAN).to(dev).eval()

lens = [len(tok(t)["input_ids"]) for t in texts]
over = sum(l > cfg.MAX_LEN for l in lens)
print(f"token length: median={int(np.median(lens))}, p90={int(np.percentile(lens, 90))}, "
      f"เกิน {cfg.MAX_LEN} = {over} คดี ({100*over/len(lens):.1f}%) -> ถูกตัดท้าย (เขียนใน limitation)")

torch.set_num_threads(os.cpu_count())
BS = 4 if dev == "cpu" else cfg.BATCH          # CPU: batch เล็กเพื่อไม่ให้ RAM ระเบิด
order = np.argsort(lens)                         # เรียงสั้น->ยาว ลด padding และ RAM
PART = P("01_partial.npz")           # checkpoint: ล้มกลางคันแล้วรันใหม่จะทำต่อจากเดิม
mean_l, cls_l, start = [], [], 0
if os.path.exists(PART):
    z = np.load(PART)
    if int(z["bs"]) == BS and int(z["n"]) <= len(texts):
        mean_l, cls_l, start = [z["mean"]], [z["cls"]], int(z["n"])
        print(f"พบ checkpoint -> ทำต่อจากคดีที่ {start}/{len(texts)}")
with torch.no_grad():
    for n, i in enumerate(range(start, len(texts), BS)):
        idx = order[i:i + BS]
        enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=cfg.MAX_LEN,
                  return_tensors="pt").to(dev)
        h = mdl(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        mean_l.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
        cls_l.append(h[:, 0].cpu().numpy())
        if n % 50 == 0:
            print(f"  {i}/{len(texts)}")
        if n % 100 == 99:
            np.savez(PART, mean=np.vstack(mean_l), cls=np.vstack(cls_l), n=i + len(idx), bs=BS)
inv = np.empty_like(order); inv[order] = np.arange(len(order))   # คืนลำดับเดิม
mean_arr = np.vstack(mean_l)[inv]; cls_arr = np.vstack(cls_l)[inv]
np.savez(OUTF, uids=df["uid"].values.astype(str), mean=mean_arr, cls=cls_arr)
if os.path.exists(PART):
    os.remove(PART)
with open(P("01_embed_info.txt"), "w", encoding="utf-8") as f:
    f.write(f"model={cfg.WANGCHAN}\nmax_len={cfg.MAX_LEN}\ntoken_median={int(np.median(lens))}\n"
            f"over_max_len={over} ({100*over/len(lens):.1f}%)\n")
print("saved", OUTF)

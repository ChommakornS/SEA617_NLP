"""ฟังก์ชันที่ใช้ร่วมกัน: โหลดข้อมูล, นิยาม relevant, bootstrap, plot"""
import collections
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg

GROUPS = ["อาญา", "แพ่ง", "แรงงาน"]
EN = {"อาญา": "Criminal", "แพ่ง": "Civil", "แรงงาน": "Labour"}   # label กราฟ (กันฟอนต์ไทยเพี้ยน)
GROUP_FILES = ["criminal", "civil", "labor"]
PENAL_GENERAL_MAX = 106   # ป.อ. ม.1-106 = ภาคทั่วไป ไม่นับเป็น relevant


def P(name):
    return os.path.join(cfg.OUT, name)


def need(path, hint):
    if not os.path.exists(path):
        sys.exit(f"ไม่พบ {path}\n-> {hint}")


def load_data():
    rows = []
    for g in GROUP_FILES:
        path = os.path.join(cfg.DATA_DIR, f"{cfg.PREFIX}_{g}.json")
        need(path, f"วางไฟล์ข้อมูลไว้ใน {cfg.DATA_DIR}/ (ดู config.py)")
        with open(path, encoding="utf-8") as f:
            rows += json.load(f)
    cols = ["uid", "case_no", "year_be", "case_group", "category", "categories",
            "principle", "laws_raw", "sections", "split"]
    df = pd.DataFrame(rows)[cols].sort_values("uid").reset_index(drop=True)
    assert df["uid"].is_unique
    return df


def clean_text(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def l2norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def load_embeddings(df, pooling):
    need(P("embeddings.npz"), "รัน 01_embed.py ก่อน")
    z = np.load(P("embeddings.npz"), allow_pickle=True)
    if list(z["uids"].astype(str)) != list(df["uid"]):
        sys.exit("ลำดับ uid ใน embeddings.npz ไม่ตรงกับข้อมูล -> ลบ results/embeddings.npz แล้วรัน 01_embed.py ใหม่")
    return l2norm(z[pooling])


# ---------------- นิยาม relevant ----------------
_SEC = re.compile(r"^(?P<law>.+?)\s+(?:ม\.|ข้อ)\s*(?P<num>[\d/]+)")


def parse_section(s):
    m = _SEC.match(s.strip())
    return (s.strip(), None) if not m else (m.group("law").strip(), m.group("num"))


def is_substantive(sec):
    """ไม่นับมาตราวิธีพิจารณา (ป.วิ.พ./ป.วิ.อ./พ.ร.บ.วิธีพิจารณา...) และ ป.อ. ภาคทั่วไป"""
    law, num = parse_section(sec)
    if law.startswith("ป.วิ.") or "วิธีพิจารณา" in law:
        return False
    if law == "ป.อ." and num and num.isdigit() and int(num) <= PENAL_GENERAL_MAX:
        return False
    return True


def substantive_sections(secs):
    return frozenset(s.strip() for s in secs if is_substantive(s))


def law_of(sec):
    return parse_section(sec)[0]


def build_eval_sets(df):
    """คืน (secs, q_idx, rel, answerable)
    query = test split | relevant = อ้างมาตราสาระร่วม >=1 | answerable = มี relevant ใน pool"""
    secs = [substantive_sections(s) for s in df["sections"]]
    q_idx = np.where(df["split"].values == "test")[0]
    inv = collections.defaultdict(set)
    for j, s in enumerate(secs):
        for x in s:
            inv[x].add(j)
    rel = {}
    for i in q_idx:
        r = set()
        for x in secs[i]:
            r |= inv[x]
        r.discard(i)
        rel[i] = r
    answerable = [i for i in q_idx if rel[i]]
    return secs, q_idx, rel, answerable


def bootstrap_ci(x, B=2000, seed=cfg.SEED):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return tuple(np.percentile(m, [2.5, 97.5]))


def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels([EN[g] for g in GROUPS])
    ax.set_yticks(range(3)); ax.set_yticklabels([EN[g] for g in GROUPS])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def banner():
    if cfg.DRY_RUN:
        print("!!! DRY_RUN = True : embedding/Typhoon เป็นของจำลอง ห้ามนำตัวเลขไปรายงาน !!!")

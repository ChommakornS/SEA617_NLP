"""STEP 7 (ไม่บังคับ): รวมไฟล์ผลลัพธ์ .csv ทั้งหมดใน results/ เป็น Excel ไฟล์เดียว -> results/07_all_results.xlsx
ต้องมี openpyxl:  python -m pip install openpyxl"""
import glob
import os

import pandas as pd

import config as cfg

files = sorted(glob.glob(os.path.join(cfg.OUT, "*.csv")))
if not files:
    raise SystemExit(f"ยังไม่มีไฟล์ .csv ใน {cfg.OUT} -> รันสคริปต์ 00-06 ก่อน")
out = os.path.join(cfg.OUT, "07_all_results.xlsx")
with pd.ExcelWriter(out, engine="openpyxl") as xw:
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0][:31]     # ชื่อ sheet ยาวได้ไม่เกิน 31 ตัวอักษร
        pd.read_csv(f, encoding="utf-8-sig").to_excel(xw, sheet_name=name, index=False)
        print("  +", name)
print("saved", out)

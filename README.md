# NLP Project Pipeline

## เตรียม
1. วางไฟล์ `deka_balanced_3000_criminal.json`, `_civil.json`, `_labor.json` ใน `data/`
2. `pip install -r requirements.txt` (ต้องเป็น `transformers>=4.51,<5` — v5 โหลด tokenizer ของ WangchanBERTa ไม่ได้)
3. แก้ค่าต่างๆ (ชื่อโมเดล, MAX_LEN, โฟลเดอร์) ที่ `config.py` ที่เดียว

## ลำดับรัน (แต่ละขั้นอ่านผลของขั้นก่อนจากโฟลเดอร์ `results/` ถ้าขั้นไหนไม่ผ่าน แก้แล้วรันเฉพาะขั้นนั้นซ้ำได้)
| ขั้น | คำสั่ง | ต้องมีก่อน | ผลที่ได้ | GPU |
|---|---|---|---|---|
| 0 | `python 00_data_summary.py` | - | ตาราง mapping 22→3, สัดส่วน % | ไม่ต้อง |
| 1 | `python 01_embed.py` | - | `01_embeddings.npz` | ควรมี |
| 2 | `python 02_classify.py` | 1 | `02_lr_preds.csv`, `02_lr_metrics.json`, confusion | ไม่ต้อง |
| 3 | `python 03_typhoon_zeroshot.py --limit 20` แล้วรันเต็ม `python 03_typhoon_zeroshot.py` | - | `03_typhoon_preds.csv`, `03_typhoon_metrics.json` | ต้องมี (หรือใช้ Colab) |
| 4 | `python 04_retrieval_eval.py` | 1, 2 (และ 3 สำหรับ Pipeline B) | `04_retrieval_summary.csv/.md`, per-query, กราฟ | ไม่ต้อง |
| 5 | `python 05_analysis.py` | 4 | `05_paired_rq.csv`, `05_explain_A.md`, `05_qualitative.md`, รายกลุ่ม | ไม่ต้อง |
| 6 | `python 06_final_tables.py` | 2, 4 (และ 3) | `06_final_tables.md` | ไม่ต้อง |

- ขั้น 3 ข้ามได้ (จะไม่มี Pipeline B) แล้วกลับมารันภายหลัง จากนั้นรันขั้น 4–6 ซ้ำ
- ขั้น 1 และ 3 เก็บผลไว้ รันซ้ำจะข้าม/รันต่อ ลบไฟล์ใน `results/` ถ้าต้องการคำนวณใหม่
- ทดสอบโดยไม่โหลดโมเดล: ตั้งตัวแปร `NLP_DRY_RUN=1` (ตัวเลขใช้ไม่ได้) แล้วรันทุกขั้น
- ใช้ `python` ปกติ (ไม่ใช้ `-I`) เพราะสคริปต์ import `common.py`/`config.py` ในโฟลเดอร์เดียวกัน

## สิ่งที่ต้องรายงานให้ตรงกับที่รันจริง
ชื่อ/รุ่น Typhoon, จำนวนคดีเกิน 510 token (`results/01_embed_info.txt`), จำนวน query ที่มีคำตอบ (ขั้น 4 พิมพ์ให้), pooling/C ที่เลือก (`02_lr_metrics.json`)

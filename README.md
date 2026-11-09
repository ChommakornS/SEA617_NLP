# วิธีรัน (Category-Aware Retrieval of Thai Supreme Court Judgments)

## เตรียม
1. วางไฟล์ `deka_balanced_3000_criminal.json`, `_civil.json`, `_labor.json` ไว้ในโฟลเดอร์ `data/`
2. `pip install -r requirements.txt`
3. ขั้น 1 และ 3 ต้องโหลดโมเดลจาก Hugging Face และควรรันบน GPU (Colab T4 พอ)

## ลำดับรัน
```bash
python 00_data_summary.py       # ตาราง mapping หมวดย่อย->กลุ่มหลัก + สัดส่วน (feedback อาจารย์)
python 01_embed.py              # WangchanBERTa embedding (mean + CLS) -> results/embeddings.npz
python 02_classify.py           # Logistic Regression  -> lr_metrics.json, lr_preds.csv
python 03_typhoon_zeroshot.py   # Typhoon zero-shot    -> typhoon_metrics.json, typhoon_preds.csv
python 04_retrieval_eval.py     # Pipeline P0/A/B/C    -> retrieval_summary.md, explain_A.md
```
ทดสอบ Typhoon เร็วๆ ก่อน: `python 03_typhoon_zeroshot.py --limit 20`

## ผลลัพธ์ที่ได้ (ใน `results/`)
| ไฟล์ | ใช้ใน |
|---|---|
| `data_summary.md`, `mapping_category_to_group.csv` | หัวข้อ Dataset / ตอบ feedback |
| `lr_metrics.json`, `lr_report.txt`, `lr_confusion.csv` | ตารางผลการจำแนก |
| `typhoon_metrics.json`, `typhoon_confusion.csv` | ตารางผลการจำแนก (baseline) |
| `retrieval_summary.md` | ตาราง Precision@K ของทุก pipeline |
| `retrieval_per_query.csv` | วิเคราะห์เพิ่ม/แยกรายหมวด |
| `explain_A.md` | reverse engineering (กฎหมายที่ถูกอ้างบ่อยใน Top-K) |

## สิ่งที่ต้องรายงานให้ตรงกับที่รันจริง
- ชื่อ/revision ของโมเดล Typhoon ที่ใช้ (แก้ `--model` ได้)
- จำนวนคดีที่ยาวเกิน 512 token (01_embed.py พิมพ์ให้)
- จำนวน test query ที่ "มีคำตอบ" ใน pool (04 พิมพ์ให้)
- ปรับนิยามมาตราที่ไม่นับเป็น relevant ได้ใน `common.py` (`is_substantive`)

## หมายเหตุ
- `uid` เช่น `train-25931` ใช้ชื่อ split เดิมของคอร์ปัส ไม่ใช่ split ปัจจุบัน ให้ดูคอลัมน์ `split` แทน
- ใช้ `python` ปกติ (ไม่ใช้ `-I`) เพราะสคริปต์ import `common.py` ในโฟลเดอร์เดียวกัน
- สคริปต์ 00, 02, 04 ทดสอบกับข้อมูลจริงแล้ว (ใช้ embedding จำลองแทน WangchanBERTa) ส่วน 01 และ 03 ต้องโหลดโมเดลจาก Hugging Face ซึ่งยังไม่ได้รันจริง ถ้า error ให้ส่ง traceback มาได้

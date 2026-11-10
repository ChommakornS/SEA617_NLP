"""ตั้งค่ากลาง: แก้ที่นี่ที่เดียว ทุกสคริปต์ใช้ร่วมกัน"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))   # โฟลเดอร์ที่ไฟล์ .py อยู่ (ไม่ขึ้นกับว่ารันจากที่ไหน)
DATA_DIR = os.path.join(_HERE, "data")                     # โฟลเดอร์ที่มี deka_balanced_3000_{criminal,civil,labor}.json
PREFIX = "deka_balanced_3000"
OUT = os.path.join(_HERE, "results")
print("[config] results จะถูกเก็บที่:", OUT)

WANGCHAN = "airesearch/wangchanberta-base-att-spm-uncased"
TYPHOON = "typhoon-ai/typhoon-s-4b-nitibench-ccl-legal-agent-research-preview"

MAX_LEN = 510     # WangchanBERTa max_position_embeddings = 512 -> ใช้ได้จริง 510 token
BATCH = 16
SEED = 42
KS = (5, 10, 20)
N_EXPLAIN = 10    # จำนวน query ตัวอย่างสำหรับ reverse engineering
N_QUAL = 5        # จำนวนตัวอย่างต่อฝั่งของ qualitative analysis

# ทดสอบ pipeline โดยไม่โหลดโมเดล (embedding/Typhoon เป็นของจำลอง ตัวเลขใช้ไม่ได้!)
#   Windows (cmd):  set NLP_DRY_RUN=1      PowerShell: $env:NLP_DRY_RUN="1"
DRY_RUN = os.environ.get("NLP_DRY_RUN", "0") == "1"

os.makedirs(OUT, exist_ok=True)

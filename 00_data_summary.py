"""STEP 0: สรุป dataset + ตาราง mapping หมวดย่อย (22) -> กลุ่มหลัก (3)  [ตอบ feedback อาจารย์]"""
import pandas as pd

from common import GROUPS, P, load_data

df = load_data()
t1 = pd.crosstab(df["case_group"], df["split"]).reindex(GROUPS)[["train", "val", "test"]]
t1["รวม"] = t1.sum(axis=1)
t1["สัดส่วน %"] = (100 * t1["รวม"] / t1["รวม"].sum()).round(1)

t2 = pd.crosstab(df["category"], df["case_group"]).reindex(columns=GROUPS, fill_value=0)
t2["รวม"] = t2.sum(axis=1)
t2["กลุ่มหลัก"] = t2[GROUPS].idxmax(axis=1)
t2 = t2.sort_values(["กลุ่มหลัก", "รวม"], ascending=[True, False])
mism = int((df["category"].map(t2["กลุ่มหลัก"]) != df["case_group"]).sum())

t1.to_csv(P("00_summary_group_split.csv"), encoding="utf-8-sig")
t2.to_csv(P("00_mapping_category_to_group.csv"), encoding="utf-8-sig")
with open(P("00_data_summary.md"), "w", encoding="utf-8") as f:
    f.write(f"# Dataset summary\n\nจำนวนคดี: {len(df)}\n\n## กลุ่ม x split\n\n{t1.to_markdown()}\n\n"
            f"## หมวดย่อย -> กลุ่มหลัก\n\n{t2.to_markdown()}\n\n"
            f"คดีที่หมวดย่อยไม่ตรงกับกลุ่มหลักของหมวดนั้น: {mism} ({100*mism/len(df):.1f}%)\n")
print(open(P("00_data_summary.md"), encoding="utf-8").read())

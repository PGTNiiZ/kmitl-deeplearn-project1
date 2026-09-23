# Unseen-writer test set

ชุดนี้ต้องแยกจาก `data/raw` และ split เดิมโดยเด็ดขาด ห้ามนำภาพหรือผลคะแนนจากชุดนี้
กลับไปเลือก hyperparameter หลายรอบ เพราะจะทำให้ test set กลายเป็น validation set โดยปริยาย

## ชุดสาธารณะที่สร้างไว้แล้ว

ชุดสมดุล 50 ภาพ × 72 คลาสสร้างจาก Burapha-TH official test split เป็นหลัก:

```bash
python -m src.build_public_test
```

ผลลัพธ์อยู่ที่ `data/unseen_test/burapha_72/` โดย 70 คลาสมาจาก Burapha-TH,
คลาส `ๆ` มาจาก ALICE-THI และคลาส `ๅ` ตัดจากประโยคที่มี `ฤๅ` ใน BEST2019
label ใน manifest เป็นรหัส TIS (`161`–`249`) ตาม checkpoint export โดยตรง
ให้เปิด `review_55_ๅ.png` ตรวจ crop ของคลาสหลังสุดก่อนรายงานคะแนน

ตรวจและรัน checkpoint ปัจจุบันกับชุดนี้ได้ด้วย:

```bash
python -m src.audit \
  --data-dir data/unseen_test/burapha_72/images \
  --output-dir results/unseen_test/audit

python -m src.inference \
  --input data/unseen_test/burapha_72/images \
  --device cpu \
  --output results/unseen_test/burapha_predictions.csv

python -m src.unseen_test score \
  --dataset-dir data/unseen_test/burapha_72 \
  --predictions results/unseen_test/burapha_predictions.csv \
  --output results/unseen_test/burapha_score.json
```

ผลที่สร้างไว้ในเครื่องนี้คือ 3,600 ภาพถูกต้อง, ไม่มีไฟล์เสีย, ไม่มี exact duplicate
ภายในชุดหรือชนกับ train manifest และ checkpoint ปัจจุบันได้ accuracy/macro recall
`5.8611%` คะแนนนี้ควรเก็บเป็น unseen baseline; อย่าปรับโมเดลตามชุดนี้แล้วรายงานซ้ำเป็น test

## 1. สร้างแบบฟอร์ม 72 คลาส

```bash
python -m src.unseen_test make-form --output data/unseen_test/form_72.png
```

พิมพ์แบบ Actual size (100%) บน A4 ให้ผู้เขียนที่ไม่เคยอยู่ในชุดฝึกคนละหนึ่งแผ่น
ใช้ Writer ID ที่ไม่ระบุตัวบุคคล เช่น `w001` และขอความยินยอมก่อนเก็บภาพ

## 2. ถ่ายและตัดภาพ

ถ่ายให้เห็นจุดดำทั้งสี่มุมของตาราง เปิดภาพแล้วจดพิกัดตามลำดับ
บนซ้าย → บนขวา → ล่างขวา → ล่างซ้าย จากนั้นรัน:

```bash
python -m src.unseen_test crop \
  --input photos/w001.jpg \
  --corners "143,391 2874,442 2820,3980 96,3912" \
  --writer-id w001 \
  --capture-id daylight
```

ผลลัพธ์อยู่ใน `data/unseen_test/collected/`:

- `images/<label>/<writer>__<capture>.png` — ภาพตัวอักษรพร้อมใช้กับ inference
- `manifest.csv` — ground truth, writer, capture และ SHA-256
- `pages/` — หน้าที่แก้มุมมองแล้วและ metadata สำหรับตรวจงานตัด
- `report.json` — จำนวนภาพ ผู้เขียน คลาส และ min/max ต่อคลาส

เปิดภาพใน `pages/` และสุ่มตรวจ crop โดยเฉพาะสระ/วรรณยุกต์ก่อนประเมินโมเดล
เป้าหมาย 50 ผู้เขียน × 72 คลาส = 3,600 ภาพ และ `minimum_per_class = maximum_per_class = 50`

## 3. รันโมเดลและสรุปคะแนน

ส่งเฉพาะโฟลเดอร์ `images/` เข้า inference (อย่าส่งทั้ง `collected/` เพราะมีภาพเต็มหน้าใน `pages/`):

```bash
python -m src.inference \
  --input data/unseen_test/collected/images \
  --device cpu \
  --output results/unseen_test/predictions.csv

python -m src.unseen_test score \
  --dataset-dir data/unseen_test/collected \
  --predictions results/unseen_test/predictions.csv \
  --output results/unseen_test/score.json
```

คำสั่งสุดท้ายสร้าง `score.json` (accuracy, macro recall, รายคลาส, รายผู้เขียน)
และ `score.csv` สำหรับดูตัวอย่างที่ทายผิด

## กติกาเพื่อให้เรียกว่า unseen test ได้จริง

1. ผู้เขียนทุกคนต้องไม่อยู่ใน train/validation และหนึ่ง Writer ID ต้องแทนคนเดียวเสมอ
2. เก็บ capture หลักเพียงหนึ่งภาพต่อผู้เขียนเพื่อไม่ให้นับลายมือเดิมซ้ำเป็นตัวอย่างอิสระ
3. กำหนดสภาพถ่ายไว้ล่วงหน้า เช่น daylight 20 คน, indoor 20 คน, shadow/tilt 10 คน
4. ตรวจ label/crop ได้ แต่ห้ามแก้เฉพาะภาพที่โมเดลทายผิด
5. รายงานทั้ง accuracy, macro-F1, per-class recall และ confusion matrix
6. Synthetic font/noise ใช้เป็น stress set หรือข้อมูลฝึกเท่านั้น ไม่ใช้แทนคะแนน unseen handwriting หลัก

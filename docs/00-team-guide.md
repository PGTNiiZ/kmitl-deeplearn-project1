# คู่มือใช้โปรเจกต์และทำงานร่วมกัน

เอกสารนี้เป็นจุดเริ่มต้นสำหรับสมาชิกทุกคน ทำตามลำดับได้เลยโดยไม่ต้องอ่านเอกสารอื่นก่อน

```text
เตรียมเครื่อง → รับข้อมูลและ split กลาง → ทดลองโมเดล → ส่งผลให้ทีม → เลือกโมเดลสุดท้าย
```

ทุกคนต้องใช้ข้อมูลและ validation split ชุดเดียวกัน เพื่อให้คะแนนเปรียบเทียบกันได้

## ส่งงานให้เพื่อนแบบไหน

แบ่งเป็น 2 ส่วน อย่าเอาทุกอย่างใส่ Git:

| ส่งผ่าน GitHub | ส่งผ่าน Drive กลาง |
|---|---|
| `src/`, `notebooks/`, `configs/`, `docs/`, tests และ requirements | dataset `clean_32x32/`, `data/splits/clean_32x32/`, checkpoints ที่เข้ารอบ และผลทดลองที่ต้องรวม |

ผู้ดูแลโปรเจกต์ push โค้ดขึ้น repository `PGTNiiZ/kmitl-deeplearn-project1`
ส่วนผู้ดูแลข้อมูลอัปโหลดแพ็กข้อมูลกลางที่มีโครงสร้างนี้:

```text
team-data/
├── clean_32x32/               # 72 คลาส
└── splits/clean_32x32/
    ├── train.csv
    ├── val.csv
    ├── label_to_index.json
    └── split_meta.json
```

วาง `clean_32x32/` ที่ `data/raw/clean_32x32/` และ split กลางทั้ง 4 ไฟล์ที่
`data/splits/clean_32x32/` ไฟล์ข้อมูลถูก ignore และต้องไม่ push เข้า GitHub

## เริ่มเทรนภายใน 10 นาที

```bash
git clone https://github.com/PGTNiiZ/kmitl-deeplearn-project1.git
cd kmitl-deeplearn-project1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-training.txt
python -m unittest discover -s tests -v
python -m jupyterlab
```

Windows PowerShell ใช้ `.venv\Scripts\Activate.ps1` แทน `source .venv/bin/activate`

เปิด `notebooks/02-model-family-search-lab.ipynb` แล้วตรวจ cell แรก:

```python
MODE = "quick"
DATA_PATH = "data/raw/clean_32x32"
SPLIT_PATH = "data/splits/clean_32x32"
POOL = "all"
ONLY_FAMILIES = ["ชื่อ family ที่ได้รับมอบหมาย"]
INCLUDE_CUSTOM_CNN = False
OUTPUT_ROOT = "results/team/ชื่อเล่น"
```

คนที่รับผิดชอบ baseline เท่านั้นตั้ง `INCLUDE_CUSTOM_CNN = True` คนอื่นตั้ง `False`
จากนั้น Restart kernel และรันตั้งแต่ต้น ต้องเห็นข้อความนี้ก่อนเริ่มเทรน:

```text
✅ REAL DATA พร้อมใช้งาน
Dataset root: .../data/raw/clean_32x32
โฟลเดอร์ภาพจริง: .../data/raw/clean_32x32
จำนวนภาพ / คลาส: ... / 72
```

หากเห็น `synthetic_0` หรือ `ชุดทดลอง: smoke` ให้หยุด เพราะกำลังใช้ผลจำลองหรือ
output เก่า ให้ปิด notebook โดยไม่บันทึก output เก่า เปิดไฟล์ใหม่ Restart kernel
แล้วรันจาก cell แรก

## แบ่ง 40 families ให้ 6 คน

ใส่รายการของตัวเองลง `ONLY_FAMILIES` เท่านั้น แต่ละคนใช้ `MODE`, image size,
batch size, seed และ split เดียวกัน:

| คน | `ONLY_FAMILIES` |
|---|---|
| A + baseline | `['ResNet', 'ResNeXt', 'DenseNet', 'DLA', 'Res2Net']` |
| B | `['ResNeSt', 'SE-ResNet', 'ECA-ResNet', 'EfficientNet', 'EfficientNetV2', 'RegNet', 'ReXNet']` |
| C | `['MixNet', 'MobileNetV2', 'MobileNetV3', 'MobileNetV4', 'GhostNet', 'MobileOne', 'RepVGG']` |
| D | `['RepViT', 'ConvNeXt', 'ConvNeXtV2', 'InceptionNeXt', 'FasterNet', 'StarNet', 'RDNet', 'HRNet']` |
| E | `['ViT', 'DeiT', 'Swin-Transformer', 'PVTv2', 'TinyViT', 'EfficientViT-MIT']` |
| F | `['MaxViT', 'MobileViT', 'EdgeNeXt', 'FastViT', 'PoolFormer', 'ConvMixer', 'CAFormer']` |

เริ่มด้วย `quick` เพื่อ screening ก่อน ผู้รวมผลจึงเลือก top candidates ไปยืนยัน
ด้วย `full` ไม่ควรให้ทุกคนรัน HPO หรือ confirmation ของทุก family ตั้งแต่รอบแรก

หลังจบ ให้แต่ละคนส่งโฟลเดอร์ `OUTPUT_ROOT` ของตัวเองผ่าน Drive โดยอย่างน้อยต้องมี
`protocol.json`, `candidate_catalog.csv`, `screening.csv`, `screening_status.csv`
และโฟลเดอร์ `runs/` ไม่ส่งคะแนนที่พิมพ์เองจากหน้าจอ

## 1. เตรียมโปรเจกต์ครั้งแรก

เปิด Terminal ที่โฟลเดอร์โปรเจกต์ แล้วรัน:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-training.txt
```

Windows PowerShell ใช้ `.venv\Scripts\Activate.ps1` แทนคำสั่ง `source`

ถ้าใช้ VS Code ให้เลือก `Python: Select Interpreter` แล้วเลือก Python ใน `.venv`

เช็กว่าติดตั้งสำเร็จ:

```bash
python -c "import torch, timm, optuna; print(torch.__version__)"
```

## 2. เตรียมข้อมูล

ถ้ายังไม่มี `.env` ให้คัดลอกไฟล์ตั้งค่า แล้วดาวน์โหลดข้อมูลกลาง:

```bash
cp -n .env.example .env
python -m src.download_data
```

ข้อมูลหลักคือ `data/raw` ถ้ามีข้อมูลอยู่ที่อื่น ให้ใส่ absolute path ใน `.env`:

```text
THAI_CHAR_DATA_DIR=/absolute/path/to/clean_32x32
```

ข้อมูลต้องมีโฟลเดอร์คลาส:

```text
clean_32x32/
└── consonants/
    ├── 0_ก/
    └── ...
```

ห้าม commit รูปภาพ, `.env`, weights หรือผลเทรนลง Git

## 3. ทดสอบว่าเครื่องพร้อม

เปิด Jupyter:

```bash
python -m jupyterlab
```

เปิด `notebooks/01-model-search-lab.ipynb` เลือก kernel `.venv` ตั้ง
`MODE = "smoke"` แล้วกด **Run All**

ถ้ารันถึง cell สุดท้ายโดยไม่ error แปลว่า environment, trainer และ inference
ทำงานครบ โหมดนี้ใช้ภาพจำลอง จึงห้ามนำคะแนนไปใส่รายงาน

ถ้ายังไม่คุ้นกับ CNN ให้รัน
[`notebooks/00-cnn-concepts-from-class.ipynb`](../notebooks/00-cnn-concepts-from-class.ipynb)
ก่อน Notebook นี้อธิบายเนื้อหา Week 2–7 ด้วยตัวอย่างเล็ก ๆ แล้วชี้ไปยังโค้ดจริง

## 4. ล็อก split กลางก่อนเริ่มทดลอง

ขั้นตอนนี้ให้ผู้รับผิดชอบข้อมูลทำเพียงคนเดียว:

1. เปิด `notebooks/01-model-search-lab.ipynb`
2. เปลี่ยน `MODE` เป็น `quick`
3. ตรวจ `DATA_PATH`, `EXPECTED_CLASSES` และ `LABEL_LEVEL`
4. รันถึงส่วน “แบ่ง train / validation”
5. ตรวจว่ามี 72 คลาส และไม่มี corrupt image หรือ label error
6. ส่งโฟลเดอร์ `data/splits/clean_32x32` ให้สมาชิกทุกคนผ่าน Drive กลาง

โฟลเดอร์ split ต้องมีครบ:

```text
data/splits/clean_32x32/
├── train.csv
├── val.csv
├── label_to_index.json
└── split_meta.json
```

สมาชิกทุกคนวางทั้ง 4 ไฟล์ไว้ที่ `data/splits/clean_32x32` ห้ามสร้าง split ใหม่เอง หากระบบ
แจ้งว่า dataset หรือ split เปลี่ยน ให้รับไฟล์กลางจากผู้รับผิดชอบข้อมูลอีกครั้ง

## 5. ทดลองโมเดลของตัวเอง

คัดลอก config ตัวอย่าง:

```bash
cp configs/experiments/resnet18_base.json configs/experiments/resnet18_mint.json
```

เปลี่ยน `resnet18_mint.json` ให้เป็นชื่อโมเดลและชื่อคนทดลองของแต่ละคน จากนั้นแก้
config โดยเปลี่ยนเฉพาะค่าที่ต้องการทดลอง ชื่อ architecture ที่ใช้ได้ เช่น
`resnet18`, `efficientnet_b0` และ `mobilenetv3_large_100`

เทรนจาก Terminal:

```bash
python -m src.train \
  --config configs/experiments/resnet18_mint.json \
  --split-dir data/splits/clean_32x32
```

คำสั่งนี้อ่านตำแหน่ง dataset จาก `.env` หากต้องใช้ path อื่นเฉพาะครั้ง ให้เพิ่ม
`--data-dir` ตามด้วย absolute path

สำหรับแบ่ง model family ทั้งหมดให้ทีม 6–7 คน ใช้ไฟล์สมาชิกจาก
[ตารางแบ่งงาน](../notebooks/team/README.md) หรือคัดลอก
`notebooks/03-team-model-template.ipynb` แล้วแก้ส่วนนี้:

```python
OWNER = "ชื่อคุณ"
TEAM_SIZE = 7  # 6 หรือ 7 ต้องตรงกันทั้งทีม
MEMBER_ID = 1  # แต่ละคนใช้หมายเลขไม่ซ้ำกัน
MODE = "quick"  # screening ด้วยข้อมูลจริง
```

Template จะเลือกหนึ่งงานต่อการ Run All ตาม `MEMBER_ID` และ `ROUND`; ส่งโฟลเดอร์ผล
พร้อม `handoff.json` ให้ผู้รวมผลก่อนเลือก shortlist และจูนต่อ

หนึ่งการทดลองควรเปลี่ยนทีละอย่าง เช่น เปลี่ยน architecture อย่างเดียว หรือเปิด
augmentation อย่างเดียว จะได้อธิบายได้ว่าคะแนนเปลี่ยนเพราะอะไร

## 6. ส่งผลกลับทีม

หลังเทรนเสร็จ โฟลเดอร์ run จะมีไฟล์สำคัญ:

```text
run_receipt.json       สรุป run และตำแหน่ง artifact
config.json            config ที่ใช้จริง
metrics.json           accuracy, macro F1 และ per-class metrics
history.csv            ผลแต่ละ epoch
confusion_matrix.csv   คลาสที่สับสนกัน
predictions.csv        ผลทำนาย validation รายภาพ
best.pt                checkpoint ที่ดีที่สุด
```

ส่งอย่างน้อย `run_receipt.json`, `config.json` และ `metrics.json` ให้คนรวมผล
ส่วน `best.pt` ส่งเฉพาะโมเดลที่เข้ารอบ อย่าพิมพ์คะแนนจากหน้าจอเอง ให้ใช้ค่าจาก
`run_receipt.json` เสมอ

## 7. กติกาเปรียบเทียบผล

ผลใน leaderboard เดียวกันต้องใช้:

- `clean_32x32/` และ `data/splits/clean_32x32/` ชุดเดียวกัน
- image size และ validation preprocessing เดียวกัน
- seed และ epoch budget ตามรอบที่ทีมกำหนด
- macro F1 เป็นคะแนนหลัก และ accuracy เป็นคะแนนประกอบ
- checkpoint ที่เลือกจาก validation macro F1 ไม่ใช่ training accuracy

ผลจาก `smoke`, `quick`, `full` หรือ split คนละชุดต้องแยกตารางกัน

## 8. ทำงานร่วมกันผ่าน Git

ก่อนเริ่มงาน ให้ดึงโค้ดล่าสุดและสร้าง branch ของตัวเอง:

```bash
git switch main
git pull
git switch -c experiment/mint-efficientnet
```

ตัวอย่าง: `experiment/mint-efficientnet`, `fix/nina-data-audit` หรือ
`docs/team-presentation`

บันทึกและส่ง branch:

```bash
git status
git add configs/experiments/resnet18_mint.json
git commit -m "ทดลอง EfficientNet-B0 ด้วย split กลาง"
git push -u origin experiment/mint-efficientnet
```

ก่อนเปิด Pull Request:

1. ล้าง output ของ notebook เพื่อลดขนาดไฟล์และ merge conflict
2. เช็กว่าไม่ได้ add `.env`, dataset, weights หรือ `results/`
3. รัน `python -m unittest discover -s tests -v`
4. เขียนว่าเปลี่ยนอะไร ทดลองอะไร และผลอยู่ที่ไหน
5. ให้เพื่อนอย่างน้อยหนึ่งคน review ก่อน merge

อย่าให้หลายคนแก้ `01-model-search-lab.ipynb` พร้อมกัน แต่ละคนควรใช้ config หรือ
notebook สำเนาของตัวเอง การแก้โค้ดกลางใน `src/` ต้องแจ้งทีมก่อน เพราะมีผลต่อทุก run

## 9. แบ่งงานในทีม

| งาน | หน้าที่ | สิ่งที่ส่ง |
|---|---|---|
| ข้อมูล | audit และล็อก split | audit report และ `data/splits` |
| Baseline | เทรน Custom CNN | ผล E0 |
| Backbone | ทดลอง ResNet/EfficientNet/MobileNet | ตาราง E1 |
| Robustness | ทดลอง augmentation และ class weights | ผล ablation |
| HPO | จูนโมเดลที่ชนะด้วย Optuna | trials และ top configs |
| Error analysis | ดู confusion และภาพที่ทายผิด | คู่คลาสที่สับสนและข้อเสนอ |
| Final/demo | เทรนยืนยันและทดสอบ inference | checkpoint และ leaderboard |

ทุกคนควรอธิบายได้ว่าข้อมูลเข้าโมเดลอย่างไร เลือก checkpoint อย่างไร และทำไมต้องดู macro F1

## 10. ปัญหาที่พบบ่อย

**หา dataset ไม่เจอ** — ตรวจ `THAI_CHAR_DATA_DIR` ใน `.env` ว่าเป็น absolute path

**จำนวนคลาสไม่ใช่ 72** — ตรวจ `DATA_PATH` และปรับ `LABEL_LEVEL` ให้ตรงกับโฟลเดอร์

**split หรือ dataset เปลี่ยน** — อย่าลบไฟล์เพื่อข้าม error ให้รับ split กลางใหม่

**CUDA out of memory** — ลด `BATCH_SIZE` เช่น 32 เป็น 16 แล้วเริ่มเทียบทุกโมเดลใหม่

**import ไม่ได้ใน notebook** — เลือก kernel `.venv` หรือ restart หลังติดตั้ง dependencies

**โหลด pretrained weights ไม่ได้** — ต่ออินเทอร์เน็ตในรอบแรก หรือใช้ cache ของทีม

## ไฟล์สำคัญ

| ไฟล์ | ใช้ทำอะไร |
|---|---|
| [`notebooks/00-cnn-concepts-from-class.ipynb`](../notebooks/00-cnn-concepts-from-class.ipynb) | เรียนแนวคิดจาก Week 2–7 ก่อนทดลองจริง |
| [`notebooks/01-model-search-lab.ipynb`](../notebooks/01-model-search-lab.ipynb) | ทำ flow ทั้งระบบและหา config ที่ดีที่สุด |
| [`notebooks/03-team-model-template.ipynb`](../notebooks/03-team-model-template.ipynb) | template แบ่ง 40 family + baseline ให้ทีม 6–7 คน |
| [`src/train.py`](../src/train.py) | เทรน ประเมิน และบันทึก checkpoint |
| [`src/search.py`](../src/search.py) | เทียบโมเดล ทำ ablation และจูน Optuna |
| [`src/inference.py`](../src/inference.py) | ทำนายภาพด้วย checkpoint |
| `data/splits/` | split และ label mapping กลางของทีม |
| `results/` | ผลทดลองในเครื่อง ไม่ commit เข้า Git |
| [`docs/04-experiments.md`](04-experiments.md) | สรุปผลทดลองที่ทีมยอมรับ |

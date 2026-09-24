# Notebooks

ลำดับใช้งาน: **00 แนวคิด CNN → 01 ทดลองโมเดล → 02 ทดลอง model family → 03 template สมาชิก**
เลขเริ่มใหม่ในโฟลเดอร์นี้; รายการ notebook 04–12 ท้ายเอกสารเป็นแผนที่ยังไม่ได้สร้าง

ถ้าต้องการทำนายภาพโดยไม่เทรน เปิด [inference-demo.ipynb](inference-demo.ipynb)
เลือก kernel `.venv`, ใส่ path รูปหรือโฟลเดอร์ใน `INPUT_PATH` แล้วกด Run All
โน้ตบุ๊กใช้โมเดล DenseNet121 E1 candidate ที่อยู่ใน repo; รูปเดี่ยวแสดง Top-3
และโฟลเดอร์บันทึก `results/predictions/notebook_predictions.csv`

สมาชิกใหม่ควรอ่าน [คู่มือใช้โปรเจกต์และทำงานร่วมกัน](../docs/00-team-guide.md)
ก่อน คู่มือระบุขั้นตอนติดตั้ง การรับ split กลาง การส่งผล และ Git workflow
ส่วนเอกสารนี้เก็บรายละเอียดเฉพาะ notebook

## เรียนแนวคิดก่อนเริ่มทดลอง

เปิด [00-cnn-concepts-from-class.ipynb](00-cnn-concepts-from-class.ipynb)
เพื่อทบทวนเนื้อหา Week 2–7 และดูว่า image array, convolution, padding,
activation, pooling, loss, backpropagation, transfer learning, augmentation
และ metrics เชื่อมกับโค้ดในโปรเจกต์อย่างไร Notebook นี้รันบน CPU ได้และไม่ต้อง
ใช้ dataset จริง

## ทดลอง model family เพิ่มจากรอบเดิม

เปิด **[02-model-family-search-lab.ipynb](02-model-family-search-lab.ipynb)** เพื่อทดลอง
40 ตัวแทน + Custom CNN พร้อมเหตุผลภาษาไทยที่เชื่อมกับแนวคิดในชั้นเรียน
มีตาราง RUN/LATER/SKIP ครบ 148 ชื่อจาก Master List, ตรวจ pretrained tag และ
forward/backward ก่อนฝึก, บันทึก failures, screening และยืนยัน top candidates หลาย seed
ใช้ trainer และ split กลางเดิม ผลแยกไว้ใน `results/model_family_search/`

ค่าเริ่มต้นใช้ `MODE="quick"` กับ `data/raw`; `smoke` ใช้เฉพาะตรวจระบบ
และห้ามนำคะแนนไปเทียบผลจริง ใช้ `POOL="all"` เพื่อรันทั้ง 40 ตัว (`core` = 20 ตัว) ข้อมูลจริงต้องมี split กลางครบ
4 ไฟล์ก่อน ตั้ง `ONLY_FAMILIES` เพื่อแบ่งงานได้ ส่วน `RUN_HPO=True` เปิดการจูน
เฉพาะ family ที่ชนะเพิ่มเติม ผลยังเป็น validation ไม่ใช่ unseen-test performance
หากบางตัวล้มเหลว notebook จะเก็บผลตัวอื่นต่อ แต่หยุดก่อนเลือกผู้ชนะจนแก้แล้ว retry
หรือเลือก `ALLOW_PARTIAL=True` เพื่อระบุว่าผลครอบคลุมเฉพาะชุดที่สำเร็จ

Notebook 02 ใช้ path จาก `.env` ได้เมื่อปล่อย `DATA_PATH` ว่าง และดาวน์โหลดจาก Drive
ให้อัตโนมัติเมื่อไม่มีข้อมูล (`DOWNLOAD_IF_MISSING=True`) ผลข้อมูลจริงอยู่ใต้ `quick/` หรือ `full/`
ไฟล์ใต้ `smoke/` เป็นผลทดสอบเก่า ไม่ใช่คะแนนของ dataset ที่ดาวน์โหลดมา

## Lab เดิม: baseline → ablation → HPO

ทั้ง `01-model-search-lab` และ `02-model-family-search-lab` มีส่วน **ดูผลด้วยตา**
หลัง screening: ภาพชุดเดียวกันเทียบทุกโมเดล พร้อมเฉลย คำทำนาย confidence และถูก/ผิด
เปิดรายละเอียดรายโมเดลเพื่อดูภาพทายผิดที่มั่นใจสูง ภาพที่ไม่มั่นใจ recall รายคลาส
และคู่คลาสสับสน หลัง export มีภาพต้นฉบับเทียบ input ที่โมเดลเห็น พร้อม Top-3 จริง
ตั้ง `PREDICT_IMAGES` เพื่อลองภาพใหม่ และ `IMAGE_TRUTH` เพื่อใส่เฉลยถ้ามี
รายงานเก็บเป็น HTML ที่ฝังภาพพร้อมเปิดดู และ CSV ใน `screening_visuals/` กับ `final_visuals/`
หากมีผลเดิมแล้ว ตั้ง `REVIEW_TABLE` เป็น path ของ CSV ผลทดลอง แล้วรันเฉพาะ
imports, data/split และ cell ดูผล ไม่ต้องเทรนใหม่ ใช้ `DISPLAY_LABELS` แปลรหัสคลาส
เป็นตัวอักษรที่ตรวจสอบแล้วได้

Open **[01-model-search-lab.ipynb](01-model-search-lab.ipynb)** for the complete
audit → frozen split → baseline/backbone benchmark → augmentation/weighting
ablation → persistent Optuna search → full-budget confirmation → inference flow.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-training.txt
.venv/bin/python -m jupyterlab
```

Select the `.venv` kernel in VS Code, or open the notebook from JupyterLab.
The notebook now defaults to `data/raw/clean_32x32` (72 classes)
and `MODE="quick"`; change `DATA_PATH` if your copy is elsewhere,
then Run All. Set `MODE="full"` when you are ready for the longer run; it uses 12 screening epochs, 12 HPO trials of 8 epochs,
then 30-epoch confirmation for two candidates plus control on two seeds.
Use `quick` for a smaller budget or `smoke` for a self-contained generated-image
test that needs no real data/pretrained download. Smoke scores are not research results.

The notebook locates the repo from either its root or `notebooks/` directory.
On Colab, clone/upload the entire repository, select a GPU and set `REPO_PATH`;
use mounted persistent storage for `OUTPUT_ROOT` and `SPLIT_PATH` to preserve
weights, the split and Optuna SQLite across sessions. The configured ZIP is
extracted automatically. Private Drive data can be mounted manually.

Outputs are under `results/model_search/<mode>/<split-hash>/`: screening CSVs,
Optuna SQLite/trials, confirmation leaderboard, curves, errors and an `export/`
folder with `best_config.json`, `best_model.pt`, mapping and split metadata.
Completed runs can be reused; interrupted training restarts from epoch one.
Exact duplicates are grouped, but near-duplicate/writer/font grouping still
requires dataset-specific review. Fixed ImageNet normalization is this lab's
comparison protocol, distinct from the original planning YAML.

Add a supported timm model name to `ARCHITECTURES` to extend the benchmark.
Keep image size, normalization, split, seeds and budgets identical when comparing
teammates' results. Do not run concurrent notebooks against one output directory.
`src.train` accepts the lab's flat **JSON**, not `configs/first_experiment.yaml`.

## Template แบ่งงานทีม 6–7 คน

[03-team-model-template.ipynb](03-team-model-template.ipynb) เป็น template สำหรับ
เทรนหนึ่ง family ต่อคนต่อรอบด้วยข้อมูลจริง มีไฟล์พร้อมแจก
[member-01 ถึง member-07 และตารางแบ่งงาน](team/README.md)
ครอบคลุม 40 family ที่คัดไว้ใน Notebook 02 และ Custom CNN หนึ่งตัว

กรอก `OWNER` และเลือก `TEAM_SIZE` เป็น 6 หรือ 7 ให้ตรงกันทั้งทีม
ไฟล์สมาชิกตั้ง `MEMBER_ID` ไว้แล้ว ทีม 6 คนใช้ไฟล์ 01–06 แล้วเปลี่ยน `TEAM_SIZE=6` ทุกไฟล์
ทุกคนใช้ split และสูตรเดียวกัน รัน screening ก่อน แล้วส่งทั้งโฟลเดอร์ผลพร้อม
`handoff.json`, CSV, config, metrics และ checkpoint ให้ผู้รวมผลเลือก shortlist ทั้งทีม
HPO และ confirmation ทำหลังรวมผล ไม่รันแยกเลือกผู้ชนะของแต่ละคน

## Reading the shared dataset path

Start Jupyter from the repository root, then place this in the first cell of
every notebook:

```python
from src.paths import get_data_dir

DATA_DIR = get_data_dir()
print(DATA_DIR)
```

`DATA_DIR` resolves to `data/raw/clean_32x32/` by default. Place the shared
`clean_32x32/` folder in `data/raw/` (or run `python3 -m src.download_data`)
once to cache the shared Google Drive folder locally, then every notebook uses
the same code without committing an absolute personal path.

Suggested notebooks after the pipeline exists:

- `04-data-setup-audit.ipynb`
- `05-split-validation.ipynb`
- `06-baseline-cnn.ipynb`
- `07-backbone-benchmark.ipynb`
- `08-augmentation-imbalance.ipynb`
- `09-optuna-hpo.ipynb`
- `10-confusion-error-analysis.ipynb`
- `11-final-training-inference.ipynb`
- `12-presentation-figures.ipynb`

See `docs/02-team-experiment-plan.md` for the owner, inputs, and required outputs
of each notebook. Do not duplicate `src/` functions inside notebooks.

# โครงสร้างไฟล์และลำดับการอ่าน

เอกสารและ notebook ใช้ชื่อ `00-file-name` โดยเริ่มเลขใหม่ในแต่ละโฟลเดอร์
เลขบอกลำดับการอ่านหรือใช้งาน ส่วน `README.md`, Python modules, tests และ config
คงชื่อมาตรฐานเพื่อรองรับเครื่องมือและคำสั่งเดิม

## เอกสาร

| ลำดับ | ไฟล์ | ใช้เมื่อ |
|---|---|---|
| 00 | [คู่มือทีม](00-team-guide.md) | เริ่มติดตั้ง ทดลอง และส่งงาน |
| 01 | [แผนโปรเจกต์](01-project-plan.md) | ทำความเข้าใจแนวทางและเหตุผลทางเทคนิค |
| 02 | [แผนการทดลองของทีม](02-team-experiment-plan.md) | แบ่งงานและกำหนดการทดลอง |
| 03 | [รายการตรวจงาน](03-execution-checklist.md) | ตรวจความคืบหน้าทีละขั้น |
| 04 | [บันทึกการทดลอง](04-experiments.md) | บันทึกและเปรียบเทียบผลที่รับรองแล้ว |
| 05 | [เอกสารส่งต่องาน](05-next-model-handoff.md) | รับช่วงพัฒนาต่อ |
| 06 | [โครงสร้างไฟล์](06-file-tree.md) | ค้นหาไฟล์และดูลำดับการอ่าน |

ข้อกำหนดต้นฉบับอยู่ที่ [00-project-requirements.md](../00-project-requirements.md)

## Notebook

| ลำดับ | ไฟล์ | หน้าที่ |
|---|---|---|
| 00 | [CNN concepts](../notebooks/00-cnn-concepts-from-class.ipynb) | ทบทวนแนวคิดจากชั้นเรียน |
| 01 | [Model search lab](../notebooks/01-model-search-lab.ipynb) | ทดลอง baseline, ablation และ HPO |
| 02 | [Model family search lab](../notebooks/02-model-family-search-lab.ipynb) | ขยายการทดลองหลาย model family |
| 03 | [Team model template](../notebooks/03-team-model-template.ipynb) | คัดลอกสำหรับการทดลองของสมาชิก |

อ่านวิธีรันใน [notebooks/README.md](../notebooks/README.md)
สมาชิกที่พร้อมทดลองหลาย family เปิด notebook 02 ได้ตามคู่มือทีม

## ไฟล์ที่มีอยู่จริง

```text
kmitl-deeplearn-project1/
├── readme.md
├── 00-project-requirements.md
├── requirements.txt
├── requirements-training.txt
├── .env.example
├── configs/
│   ├── first_experiment.yaml
│   └── experiments/resnet18_base.json
├── data/                         # dataset และ split กลางในเครื่อง
│   └── README.md
├── docs/
│   ├── 00-team-guide.md
│   ├── 01-project-plan.md
│   ├── 02-team-experiment-plan.md
│   ├── 03-execution-checklist.md
│   ├── 04-experiments.md
│   ├── 05-next-model-handoff.md
│   └── 06-file-tree.md
├── notebooks/
│   ├── README.md
│   ├── 00-cnn-concepts-from-class.ipynb
│   ├── 01-model-search-lab.ipynb
│   ├── 02-model-family-search-lab.ipynb
│   └── 03-team-model-template.ipynb
├── src/
│   ├── README.md
│   ├── __init__.py
│   ├── paths.py
│   ├── download_data.py
│   ├── audit.py
│   ├── split.py
│   ├── train.py
│   ├── search.py
│   ├── inference.py
│   └── visualize.py
├── tests/
│   ├── test_audit.py
│   ├── test_paths.py
│   ├── test_split.py
│   ├── test_training.py
│   ├── test_family_notebook.py
│   └── test_visualize.py
├── results/                      # ผลทดลองที่สร้างในเครื่อง
│   └── README.md
├── weights/                      # checkpoint ที่คัดเลือก
│   └── README.md
└── presentation/
    └── README.md
```

ไฟล์ notebook 04–12 ที่กล่าวถึงในแผนการทดลองเป็นข้อเสนอสำหรับแยกงานภายหลัง
ยังไม่ได้สร้าง ไม่จำเป็นต้องสร้างไฟล์เปล่าให้ครบตามแผน

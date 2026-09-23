# Data Layout

Dataset files are intentionally not committed. Google Drive is the team source
of truth. The training dataset is `data/raw`; notebooks extract
it automatically before auditing or training.

```bash
python3 -m src.download_data
```

The command reads `THAI_CHAR_DRIVE_URL` from `.env`, which is created by copying
`.env.example`. After download, notebooks and scripts work without any personal
path configuration.

If a member uses a local cache elsewhere, they additionally set:

```text
THAI_CHAR_DATA_DIR=/their/local/or/mounted/clean_32x32
```

`.env` is ignored by Git, so each person can use a different local cache path.
All code must resolve data through `src.paths.get_data_dir()`, never by hard-
coding a personal path or `../data` in a notebook.

```text
data/
├── raw/
│   └── clean_32x32/      # immutable downloaded source used for training
├── clean/               # optional validated/canonical view
├── splits/clean_32x32/
│   ├── train.csv
│   ├── val.csv
│   ├── label_to_index.json
│   └── split_meta.json
└── synthetic/           # optional training-only generated images
```

ชุดทดสอบจากผู้เขียนใหม่ต้องอยู่นอก training split ดูขั้นตอนสร้างแบบฟอร์มและตัดภาพที่
[`docs/07-unseen-test-set.md`](../docs/07-unseen-test-set.md) ชุดนี้สร้างด้วย
`python -m src.unseen_test` และไม่ถูกอ่านโดย training pipeline อัตโนมัติ

The audit must determine the actual source layout. Do not assume a class-folder
format. Validation must contain real images only; synthetic images never enter
validation. Duplicate group IDs must stay entirely within one split.

ขั้นตอนตั้งเครื่อง รับ split กลาง และกติกาการแชร์ไฟล์อยู่ใน
[`docs/00-team-guide.md`](../docs/00-team-guide.md)

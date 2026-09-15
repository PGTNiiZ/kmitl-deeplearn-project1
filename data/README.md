# Data Layout

Dataset files are intentionally not committed. Google Drive is the team source
of truth. Each member downloads the same shared folder into `data/raw/`, the
default local cache, by running:

```bash
python3 -m src.download_data
```

The command reads `THAI_CHAR_DRIVE_URL` from `.env`, which is created by copying
`.env.example`. After download, notebooks and scripts work without any personal
path configuration.

If a member uses a local cache elsewhere, they additionally set:

```text
THAI_CHAR_DATA_DIR=/their/local/or/mounted/dataset/path
```

`.env` is ignored by Git, so each person can use a different local cache path.
All code must resolve data through `src.paths.get_data_dir()`, never by hard-
coding a personal path or `../data` in a notebook.

```text
data/
├── raw/                 # immutable downloaded source
├── clean/               # optional validated/canonical view
├── splits/
│   ├── train.csv
│   ├── val.csv
│   └── label_to_index.json
└── synthetic/           # optional training-only generated images
```

The audit must determine the actual source layout. Do not assume a class-folder
format. Validation must contain real images only; synthetic images never enter
validation. Duplicate group IDs must stay entirely within one split.

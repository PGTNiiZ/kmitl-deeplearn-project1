"""Readable, self-contained HTML reports from saved validation predictions."""
from __future__ import annotations

import base64
from html import escape
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image


STYLE = """<style>
.pred-report {font:15px/1.6 system-ui,sans-serif;color:#172033;background:#fff;padding:20px;overflow:auto}
.pred-report table {border-collapse:collapse;width:100%;margin:12px 0}
.pred-report th,.pred-report td {border:1px solid #d5dce5;padding:8px;text-align:left;vertical-align:top}
.pred-report th {background:#eef2f7;color:#172033}
.pred-report img {max-width:112px;height:96px;object-fit:contain;background:#f1f3f6;border:1px solid #d5dce5}
.pred-report .correct {background:#e5f5ea;color:#13562d}
.pred-report .wrong {background:#fff0f0;color:#952323}
.pred-report .unknown {background:#f1f3f6;color:#334155}
.pred-report summary {cursor:pointer;font-weight:650;padding:12px;background:#eef2f7}
.pred-report details {margin:16px 0}.pred-report small {display:block;overflow-wrap:anywhere}
.pred-report .cards {display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
.pred-report .card {border:1px solid #cbd5e1;border-radius:8px;padding:12px}
</style>"""


def thumbnail(image: Image.Image) -> str:
    image = image.convert("RGB")
    image.thumbnail((160, 160))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img alt="ภาพตัวอย่าง" src="data:image/png;base64,{encoded}">'


def image_html(path: Path) -> str:
    with Image.open(path) as image:
        return thumbnail(image)


def label_text(label, labels=None):
    raw = str(label)
    name = (labels or {}).get(raw, raw)
    return escape(raw if name == raw else f"{name} [{raw}]")


def outcome(actual, predicted):
    if actual is None:
        return "unknown", "ไม่มีเฉลย — ยังประเมินถูก/ผิดไม่ได้"
    return ("correct", "ถูก / Correct") if actual == predicted else ("wrong", "ผิด / Incorrect")


def load_predictions(receipt, validation_rows):
    """Join by path, never row order; fail rather than display misaligned answers."""
    table = pd.read_csv(Path(receipt["checkpoint_path"]).parent / "predictions.csv",
                        dtype={"path": str, "true_label": str, "predicted_label": str}, keep_default_na=False)
    expected = pd.DataFrame(validation_rows)[["path", "label"]].astype(str)
    if table.path.duplicated().any() or expected.path.duplicated().any():
        raise ValueError("Duplicate prediction/validation paths")
    if set(table.path) != set(expected.path):
        raise ValueError("Predictions do not cover exactly the validation manifest")
    checked = expected.merge(table, on="path", validate="one_to_one")
    if not checked.label.eq(checked.true_label).all():
        raise ValueError("Prediction ground truth differs from validation manifest")
    if not pd.to_numeric(checked.confidence, errors="coerce").between(0, 1).all():
        raise ValueError("Invalid prediction confidence")
    checked["correct"] = checked.true_label.eq(checked.predicted_label)
    return checked.drop(columns="label")


def prediction_cards(table, data_dir, labels=None):
    cards = []
    for row in table.itertuples():
        css, status = outcome(row.true_label, row.predicted_label)
        cards.append(f'<article class="card {css}">{image_html(Path(data_dir) / row.path)}'
                     f'<p><b>เฉลย:</b> {label_text(row.true_label, labels)}<br>'
                     f'<b>ทำนาย:</b> {label_text(row.predicted_label, labels)}<br>'
                     f'<b>{status}</b> · confidence {row.confidence:.1%}</p>'
                     f'<small>{escape(row.path)}</small></article>')
    return '<div class="cards">' + ''.join(cards) + '</div>' if cards else '<p>ไม่มีภาพในกลุ่มนี้</p>'


def document(title, body):
    return (f'<!doctype html><html lang="th"><meta charset="utf-8"><title>{escape(title)}</title>'
            f'{STYLE}<main class="pred-report">{body}</main></html>')


def validation_gallery(receipts, data_dir, validation_rows, output_dir, *, seed=42,
                       sample_count=8, labels=None, context="validation"):
    """One report per run plus a fair side-by-side sample across every run.

    Uses saved best-checkpoint predictions only: no GPU or model reload required.
    Returns embeddable HTML and writes standalone reports/CSV for later browsing.
    """
    if sample_count < 1 or not receipts:
        raise ValueError("Need at least one model and one sample")
    hashes = {r["split_hash"] for r in receipts}
    phases = {r["phase"] for r in receipts}
    if len(hashes) != 1 or len(phases) != 1:
        raise ValueError("Compare only the same split and evaluation phase")
    run_ids = [r["run_id"] for r in receipts]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Duplicate run IDs")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed = pd.DataFrame(validation_rows).sort_values("path").sample(
        n=min(sample_count, len(validation_rows)), random_state=seed)
    tables, sections, summary = {}, [], []
    for receipt in receipts:
        name = f'{receipt.get("family", receipt["backbone"])} · seed {receipt["seed"]}'
        table = load_predictions(receipt, validation_rows)
        run_id = receipt["run_id"]
        # Run IDs are generated by the trainer; reject external path components.
        if Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("Invalid run ID")
        table.to_csv(output_dir / f"{run_id}_answers.csv", index=False)
        tables[run_id] = table.set_index("path")
        sample = table.set_index("path").loc[fixed.path].reset_index()
        errors = table[~table.correct].sort_values(["confidence", "path"], ascending=[False, True]).head(sample_count)
        unsure = table.sort_values(["confidence", "path"]).head(sample_count)
        per_class = table.groupby("true_label").agg(support=("correct", "size"), correct=("correct", "sum"))
        per_class["recall"] = per_class.correct / per_class.support
        per_class = per_class.sort_values(["recall", "support"])
        per_class.to_csv(output_dir / f"{run_id}_per_class.csv")
        pairs = table[~table.correct].groupby(["true_label", "predicted_label"]).size().sort_values(
            ascending=False).rename("count").reset_index()
        count = int(table.correct.sum())
        heading = f'{name} — ถูก {count}/{len(table)} ({count / len(table):.1%})'
        summary.append({"run_id": run_id, "model": name, "phase": receipt["phase"],
                        "correct": count, "wrong": len(table) - count,
                        "accuracy": count / len(table), "macro_f1": receipt["val_macro_f1"]})
        body = (f'<p>ชุดทดลอง: {escape(context)}' + (' — ภาพจำลอง ทดสอบระบบเท่านั้น' if context == 'smoke' else '') + '</p>'
                f'<h2>{escape(heading)}</h2><p>phase: {escape(receipt["phase"])} · '
                f'macro F1: {receipt["val_macro_f1"]:.4f} · best epoch: {receipt["best_epoch"]}</p>'
                '<h3>1. ภาพสุ่มชุดเดียวกันทุกโมเดล</h3>' + prediction_cards(sample, data_dir, labels)
                + '<h3>2. ทายผิดแต่มั่นใจสูง — เลือกเฉพาะข้อผิดพลาด ไม่ใช่สัดส่วน accuracy</h3>'
                + prediction_cards(errors, data_dir, labels)
                + '<h3>3. ภาพที่มั่นใจต่ำ — อาจทายถูกหรือผิด</h3>' + prediction_cards(unsure, data_dir, labels)
                + '<h3>4. Recall รายคลาส (เฉลย) — เรียงคลาสที่ยังทำได้ไม่ดี</h3>'
                + per_class.to_html(escape=True, float_format=lambda x: f'{x:.3f}')
                + '<h3>5. คู่ที่สับสน: เฉลย → ทำนาย</h3>'
                + (pairs.head(20).to_html(index=False, escape=True) if len(pairs) else '<p>ไม่พบภาพทายผิด</p>'))
        (output_dir / f'{run_id}.html').write_text(document(heading, body), encoding="utf-8")
        sections.append(f'<details><summary>{escape(heading)}</summary>{body}</details>')
    comparison_rows, csv_rows = [], []
    headers = ''.join(f'<th>{escape(r.get("family", r["backbone"]))}<br>seed {r["seed"]}</th>' for r in receipts)
    for row in fixed.itertuples():
        cells = [f'<td>{image_html(Path(data_dir) / row.path)}<small>{escape(row.path)}</small></td>',
                 f'<td>{label_text(row.label, labels)}</td>']
        for receipt in receipts:
            prediction = tables[receipt["run_id"]].loc[row.path]
            css, status = outcome(str(row.label), prediction.predicted_label)
            cells.append(f'<td class="{css}">{label_text(prediction.predicted_label, labels)}<br>'
                         f'{status}<br>{prediction.confidence:.1%}</td>')
            csv_rows.append({"path": row.path, "true_label": row.label, "run_id": receipt["run_id"],
                             "predicted_label": prediction.predicted_label,
                             "confidence": prediction.confidence, "correct": prediction.correct})
        comparison_rows.append('<tr>' + ''.join(cells) + '</tr>')
    comparison = '<table><thead><tr><th>ภาพจริง</th><th>เฉลย</th>' + headers + '</tr></thead><tbody>' + ''.join(comparison_rows) + '</tbody></table>'
    pd.DataFrame(csv_rows).to_csv(output_dir / 'same_images_comparison.csv', index=False)
    pd.DataFrame(summary).to_csv(output_dir / 'model_summary.csv', index=False)
    body = ('<h1>ดูคำทำนาย validation ด้วยภาพจริง</h1>'
            f'<p>ชุดทดลอง: {escape(context)}' + (' — ภาพจำลอง ทดสอบระบบเท่านั้น' if context == 'smoke' else '') + '</p>'
            '<p>เขียว + “ถูก” / แดง + “ผิด” มีข้อความกำกับเสมอ · confidence เป็น softmax score '
            'ไม่ใช่โอกาสถูกที่ผ่าน calibration · ภาพสุ่มเป็นตัวอย่างเท่านั้น คะแนนรวมใช้ validation ทั้งชุด</p>'
            + pd.DataFrame(summary).to_html(index=False, escape=True, float_format=lambda x: f'{x:.4f}')
            + '<h2>ภาพชุดเดียวกัน ใครทายว่าอะไร?</h2>' + comparison
            + '<h2>เปิดดูรายละเอียดแต่ละโมเดล</h2>' + ''.join(sections))
    (output_dir / 'index.html').write_text(document('Validation gallery', body), encoding="utf-8")
    return STYLE + '<div class="pred-report">' + body + '</div>'


def live_prediction_gallery(predictor, paths, *, truth=None, labels=None, context="prediction"):
    """Predict new images, showing original, exact model input, top-3 and known truth."""
    import numpy as np
    truth = truth or {}
    results = predictor.predict(paths, top_k=3)
    records, cards = [], []
    for path, result in zip(paths, results):
        actual = truth.get(str(Path(path).resolve()))
        actual = str(actual) if actual is not None else None
        predicted = result["predicted_label"]
        css, status = outcome(actual, predicted)
        with Image.open(path) as image:
            tensor = predictor.transform(image.convert("RGB"))
            pixels = tensor.permute(1, 2, 0).numpy() * np.array(predictor.config.std) + np.array(predictor.config.mean)
            model_input = Image.fromarray((pixels.clip(0, 1) * 255).round().astype("uint8"))
            images = f'<div>ภาพจริง {thumbnail(image)} → ภาพเข้าโมเดล {thumbnail(model_input)}</div>'
        top3 = '<ol>' + ''.join(f'<li>{label_text(p["label"], labels)}: {p["confidence"]:.1%}</li>' for p in result['top_k']) + '</ol>'
        answer = label_text(actual, labels) if actual is not None else 'ยังไม่มีเฉลย'
        cards.append(f'<article class="card {css}">{images}<p>เฉลย: <b>{answer}</b><br>'
                     f'ทำนาย: <b>{label_text(predicted, labels)}</b><br>{status}</p>'
                     f'<b>Top-3</b>{top3}<small>{escape(str(path))}</small></article>')
        records.append({"path": str(path), "true_label": actual, "predicted_label": predicted,
                        "confidence": result["confidence"], "correct": None if actual is None else actual == predicted,
                        "top3": result['top_k']})
    body = ('<h2>Predict ภาพใหม่: ภาพ → preprocessing → โมเดล → Top-3 → เทียบเฉลย</h2>'
            f'<p>โมเดล: {escape(getattr(predictor.config, "architecture", "selected checkpoint"))}</p>'
            f'<p>ชุดทดลอง: {escape(context)} · confidence ไม่ใช่โอกาสถูกที่ผ่าน calibration</p>'
            '<div class="cards">' + ''.join(cards) + '</div>')
    return STYLE + '<div class="pred-report">' + body + '</div>', pd.DataFrame(records)

"""Offline checks for the family notebook's catalog, failure handling and full flow."""
import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "family-test-mpl"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import nbformat
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/02-model-family-search-lab.ipynb"


class FamilyNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.notebook = nbformat.read(NOTEBOOK, as_version=4)

    def cell(self, tag):
        return next(c.source for c in self.notebook.cells if tag in c.metadata.get("tags", []))

    def setup_namespace(self):
        ns = {}
        exec(self.cell("settings"), ns)
        ns["REPO_PATH"] = str(ROOT)
        exec(self.cell("imports"), ns)
        ns["display"] = lambda *args: None
        exec(self.cell("candidates"), ns)
        exec(self.cell("coverage"), ns)
        return ns

    def test_catalog_and_unavailable_weights(self):
        nbformat.validate(self.notebook)
        for cell in self.notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, "notebook-cell", "exec")
        ns = self.setup_namespace()
        self.assertEqual(len(ns["candidate_table"]), 40)
        self.assertEqual(len(ns["MASTER_FAMILIES"]), 148)
        self.assertEqual(sum(ns["coverage_table"].decision == "RUN"), 40)
        tree = ast.parse(self.cell("protocol"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        exec(compile(ast.Module(body=[function], type_ignores=[]), "resolver", "exec"), ns)
        row = ns["candidate_table"].iloc[0].to_dict()
        self.assertIn(".", ns["resolve_candidate"](row)["architecture"])
        with patch.object(ns["timm"], "is_model", return_value=False):
            self.assertEqual(ns["resolve_candidate"](row)["status"], "unavailable")
        with patch.object(ns["timm"].models, "get_pretrained_cfg", return_value=None):
            self.assertEqual(ns["resolve_candidate"](row)["status"], "unavailable")

    def test_real_data_requires_the_shared_split(self):
        source = next(c.source for c in self.notebook.cells
                      if 'if MODE == "smoke":\n    data_dir' in c.source)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ns = dict(MODE="quick", DATA_PATH=str(root / "images"), SPLIT_PATH=str(root / "split"),
                      project_dir=ROOT, configured_data_dir=lambda: root / "images",
                      EXPECTED_CLASSES=72, LABEL_LEVEL=1, Path=Path, results_dir=root / "results")
            with self.assertRaisesRegex(FileNotFoundError, "split"):
                exec(source, ns)

    def test_offline_smoke_hpo_export_reuse_and_failed_candidate(self):
        ns = self.setup_namespace()
        with tempfile.TemporaryDirectory() as directory, patch.object(plt, "show"):
            ns.update(MODE="smoke", DEVICE="cpu", RUN_HPO=True, results_dir=Path(directory),
                      budget=(2, 2), OUTPUT_ROOT=directory)
            for cell in self.notebook.cells:
                if cell.cell_type != "code":
                    continue
                tag = cell.metadata["tags"][0]
                if tag in {"settings", "imports", "candidates", "coverage"}:
                    continue
                exec(cell.source, ns)
            self.assertEqual(ns["selection_stage"], "recipe_confirmation")
            exported = json.loads((ns["export_dir"] / "selection.json").read_text())
            self.assertEqual(exported["mode"], "smoke")
            self.assertEqual(len(ns["receipts"]), 2)
            self.assertTrue((ns["review_dir"] / "index.html").is_file())
            comparison = pd.read_csv(ns["review_dir"] / "same_images_comparison.csv")
            self.assertTrue((comparison.groupby("path").run_id.nunique() == 2).all())
            # Reviewing saved artifacts must not call the trainer.
            review_source = self.cell("visual-screening").replace(
                'REVIEW_TABLE = ""', 'REVIEW_TABLE = ' + repr(str(ns["experiment_dir"] / "screening.csv")))
            with patch.dict(ns, {"fit": lambda *a, **kw: self.fail("Review retrained a model")}):
                exec(review_source, ns)
                exec(self.cell("visual-final").replace("PREDICT_MODEL_INDEX = None", "PREDICT_MODEL_INDEX = 0"), ns)
            self.assertEqual(len(ns["live_answers"]), min(8, len(ns["val_rows"])))
            run_ids = {r["run_id"] for r in ns["receipts"]}
            exec(self.cell("screening"), ns)
            self.assertEqual({r["run_id"] for r in ns["receipts"]}, run_ids)

            # A failed model must not prevent later candidates from completing.
            real_fit = ns["fit"]
            def fail_custom(config, *args, **kwargs):
                if config.architecture == "custom_cnn":
                    raise RuntimeError("simulated OOM")
                return real_fit(config, *args, **kwargs)
            ns["fit"] = fail_custom
            exec(self.cell("screening"), ns)
            self.assertEqual([r["status"] for r in ns["statuses"]], ["failed", "complete"])
            self.assertEqual(len(pd.read_csv(ns["experiment_dir"] / "screening.csv")), 1)
            with self.assertRaisesRegex(RuntimeError, "Screening"):
                exec(self.cell("confirmation"), ns)
        plt.close("all")


if __name__ == "__main__":
    unittest.main()

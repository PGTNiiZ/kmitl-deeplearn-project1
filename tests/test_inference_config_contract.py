"""Regression checks for the checkpoint's JSON export contract."""
import unittest

from src.inference import load_config


class InferenceConfigContractTest(unittest.TestCase):
    def test_json_mean_std_lists_match_checkpoint_tuples(self):
        checkpoint = {"config": {"architecture": "densenet121.ra_in1k",
                                 "mean": (0.485, 0.456, 0.406),
                                 "std": (0.229, 0.224, 0.225)}}
        # External JSON decodes tuples as lists, as in the actual DenseNet export.
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import json
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(checkpoint["config"]), encoding="utf-8")
            config = load_config(path, checkpoint)
        self.assertEqual(tuple(config.mean), checkpoint["config"]["mean"])

    def test_real_config_conflict_is_rejected(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import json
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"image_size": 224}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_config(path, {"config": {"image_size": 128}})

"""Small offline checks for checkpoint parity, pruning and model interfaces."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import optuna
import pandas as pd
import torch
from PIL import Image

from src.audit import audit_dataset, DEFAULT_EXTENSIONS
from src.inference import Predictor
from src.split import prepare_split, read_rows
from src.train import TrainConfig, build_model, fit


class TrainingIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'images'
        rng = np.random.default_rng(42)
        for label in range(2):
            folder = self.data / str(label)
            folder.mkdir(parents=True)
            for i in range(5):
                Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(folder / f'{i}.png')
        audit_dataset(self.data, self.root / 'audit', 1, DEFAULT_EXTENSIONS)
        self.split = self.root / 'splits'
        prepare_split(self.data, self.root / 'audit', self.split, expected_classes=2)
        self.config = TrainConfig(architecture='custom_cnn', pretrained=False, image_size=32,
                                  batch_size=4, epochs=2, freeze_epochs=0, device='cpu', class_weights=True)

    def test_checkpoint_reload_matches_evaluation_and_completed_run_reused(self):
        receipt = fit(self.config, self.data, self.split, self.root / 'runs')
        predicted = Predictor(Path(receipt['checkpoint_path']), device='cpu').predict(
            [self.data / row['path'] for row in read_rows(self.split / 'val.csv')])
        saved = pd.read_csv(Path(receipt['checkpoint_path']).parent / 'predictions.csv', dtype={'predicted_label': str})
        self.assertEqual([r['predicted_label'] for r in predicted], saved.predicted_label.tolist())
        np.testing.assert_allclose([r['confidence'] for r in predicted], saved.confidence, atol=1e-6)
        self.assertEqual(receipt['run_id'], fit(self.config, self.data, self.split, self.root / 'runs')['run_id'])
        checkpoint = torch.load(receipt['checkpoint_path'], weights_only=True)
        self.assertEqual(checkpoint['split_hash'], receipt['split_hash'])
        self.assertEqual(checkpoint['best_epoch'], receipt['best_epoch'])

    def test_pruned_run_is_never_complete(self):
        trial = Mock()
        trial.should_prune.return_value = True
        with self.assertRaises(optuna.TrialPruned):
            fit(self.config, self.data, self.split, self.root / 'runs', phase='hpo', trial=trial)
        receipt = json.loads(next((self.root / 'runs').glob('*/run_receipt.json')).read_text())
        self.assertEqual(receipt['status'], 'pruned')
        trial.report.assert_called_once()

    def test_timm_candidates_have_trainable_heads_and_matching_output(self):
        for name in ['resnet18', 'efficientnet_b0', 'mobilenetv3_large_100']:
            with self.subTest(architecture=name):
                model = build_model(TrainConfig(architecture=name, pretrained=False), 3)
                model.eval()
                with torch.no_grad():
                    self.assertEqual(tuple(model(torch.zeros(2, 3, 32, 32)).shape), (2, 3))
                self.assertTrue(list(model.get_classifier().parameters()))

    def test_head_warmup_transitions_to_finetuning_without_network(self):
        config = TrainConfig(architecture='resnet18', pretrained=True, image_size=32,
                             batch_size=4, epochs=2, freeze_epochs=1, device='cpu')
        # Exercise the real warmup control flow using offline initial weights.
        with patch('src.train.build_model', side_effect=lambda cfg, classes: build_model(cfg, classes, False)):
            receipt = fit(config, self.data, self.split, self.root / 'runs')
        history = pd.read_csv(Path(receipt['checkpoint_path']).parent / 'history.csv')
        self.assertEqual(history.stage.tolist(), ['head', 'finetune'])
        np.testing.assert_allclose(history.lr, [config.head_lr, config.finetune_lr])


if __name__ == '__main__':
    unittest.main()

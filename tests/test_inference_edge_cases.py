"""Focused regression checks for classifier inference edge cases."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor

from src.inference import Predictor, _tta_views, load_image_safe, print_single, save_csv
from src.train import TrainConfig, make_transform


class InferenceEdgeCasesTest(unittest.TestCase):
    def predictor(self):
        predictor = object.__new__(Predictor)
        predictor.device = torch.device('cpu')
        predictor.labels = ['ก', 'ข']
        predictor.confidence_threshold = 0.0
        predictor.config = TrainConfig(architecture='fixture', image_size=2)
        predictor.transform = make_transform(predictor.config)
        linear = torch.nn.Linear(12, 2)
        with torch.no_grad():
            linear.weight.zero_()
            linear.bias.copy_(torch.tensor([2., 1.]))
        predictor.model = torch.nn.Sequential(torch.nn.Flatten(), linear)
        predictor.model.eval()
        return predictor

    def test_top_one_has_no_margin_in_all_prediction_paths_and_exports(self):
        predictor = self.predictor()
        image = Image.new('RGB', (2, 2), 'white')
        self.assertIsNone(predictor.predict_tensor(ToTensor()(image), top_k=1).margin)
        self.assertIsNone(predictor.predict_image(image, top_k=1).margin)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.png'
            image.save(path)
            result = predictor.predict([path], top_k=1)[0]
            self.assertIsNone(result['margin'])
            self.assertEqual(len(result['top_k']), 1)
            self.assertGreater(result['confidence'], 0)
            self.assertEqual(result['status'], 'OK')
            with contextlib.redirect_stdout(io.StringIO()) as output:
                print_single(result, predictor, path, 'none')
            self.assertNotIn('Top1-Top2 margin', output.getvalue())
            self.assertIsNone(json.loads(json.dumps(result))['margin'])
            csv_path = Path(directory) / 'out.csv'
            save_csv(csv_path, [result])
            with csv_path.open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row['margin'], '')
            self.assertEqual(row['top2_label'], '')

    def test_transparent_palette_png_uses_white_pad_background(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'palette.png'
            image = Image.new('P', (2, 1))
            image.putpalette([0, 0, 0, 255, 0, 0] + [0, 0, 0] * 254)
            image.putdata([0, 1])
            image.info['transparency'] = 0
            image.save(path)
            self.assertEqual([load_image_safe(path).getpixel((x, 0)) for x in range(2)],
                             [(255, 255, 255), (255, 0, 0)])

    def test_safe_tta_is_repeatable_and_none_is_identity(self):
        image = Image.new('RGB', (9, 7), 'white')
        image.putpixel((4, 3), (0, 0, 0))
        self.assertIs(_tta_views(image, 'none')[0], image)
        first, second = _tta_views(image, 'safe'), _tta_views(image, 'safe')
        self.assertEqual(len(first), 5)
        self.assertEqual([view.tobytes() for view in first],
                         [view.tobytes() for view in second])
        self.assertEqual([view.size for view in first], [image.size] * 5)

    def test_exif_transpose_rgb_and_grayscale_and_invalid_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gray = root / 'gray.png'
            Image.new('L', (2, 2), 42).save(gray)
            self.assertEqual(load_image_safe(gray).getpixel((0, 0)), (42, 42, 42))
            rgb = root / 'rgb.webp'
            Image.new('RGB', (2, 2), (10, 20, 30)).save(rgb)
            self.assertEqual(load_image_safe(rgb).mode, 'RGB')
            rgba = root / 'rgba.png'
            Image.new('RGBA', (2, 2), (0, 0, 0, 0)).save(rgba)
            self.assertEqual(load_image_safe(rgba).getpixel((0, 0)), (255, 255, 255))
            oriented = root / 'oriented.jpg'
            exif = Image.Exif()
            exif[274] = 6
            Image.new('RGB', (3, 2), (10, 20, 30)).save(oriented, exif=exif)
            self.assertEqual(load_image_safe(oriented).size, (2, 3))
            bad = root / 'bad.png'
            bad.write_bytes(b'not an image')
            with self.assertRaisesRegex(ValueError, 'unreadable or unsupported'):
                load_image_safe(bad)

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.audit import DEFAULT_EXTENSIONS, audit_dataset
from src.split import prepare_split, read_rows, source_id_for_path


class FrozenSplitTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'images'
        for label in range(3):
            folder = self.data / str(label)
            folder.mkdir(parents=True)
            for index in range(10):
                Image.new('RGB', (20, 24), (label * 80, index * 20, 50)).save(folder / f'{index}.png')
        shutil.copyfile(self.data / '0/0.png', self.data / '0/duplicate.png')
        self.audit = self.root / 'audit'
        self.splits = self.root / 'splits'
        self.run_audit()

    def run_audit(self):
        audit_dataset(self.data, self.audit, 1, DEFAULT_EXTENSIONS)

    def split(self):
        return prepare_split(self.data, self.audit, self.splits, expected_classes=3)

    def test_group_isolation_coverage_portability_and_reuse(self):
        metadata = self.split()
        train, val = read_rows(self.splits / 'train.csv'), read_rows(self.splits / 'val.csv')
        self.assertFalse({r['sha256'] for r in train} & {r['sha256'] for r in val})
        self.assertFalse({r['source_id'] for r in train} & {r['source_id'] for r in val})
        self.assertEqual({r['label'] for r in train}, {'0', '1', '2'})
        self.assertEqual({r['label'] for r in val}, {'0', '1', '2'})
        self.assertEqual(len(train) + len(val), 31)
        self.assertTrue(all(not Path(r['path']).is_absolute() for r in train + val))
        self.assertEqual(metadata, self.split())
        # Re-auditing a relocated dataset preserves the frozen identity.
        moved = self.root / 'moved'
        shutil.copytree(self.data, moved)
        self.data = moved
        self.run_audit()
        self.assertEqual(metadata, self.split())

    def test_modified_manifest_is_rejected(self):
        self.split()
        path = self.splits / 'train.csv'
        path.write_text(path.read_text().replace('0/0.png', '0/altered.png'))
        # Ensure a definitely present path is changed even if 0/0 belongs to validation.
        rows = read_rows(path)
        rows[0]['path'] = 'tampered.png'
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['path', 'label', 'sha256', 'source_id'])
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, 'modified'):
            self.split()

    def test_changed_dataset_or_seed_is_rejected(self):
        self.split()
        with self.assertRaisesRegex(ValueError, 'changed'):
            prepare_split(self.data, self.audit, self.splits, seed=123, expected_classes=3)
        Image.new('RGB', (25, 25), (2, 3, 4)).save(self.data / '0/new.png')
        self.run_audit()
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.split()

    def test_conflicting_labels_are_quarantined_or_can_be_rejected(self):
        shutil.copyfile(self.data / '0/0.png', self.data / '1/wrong-label.png')
        self.run_audit()
        metadata = self.split()
        self.assertEqual(metadata['excluded_conflicting_hashes'], 1)
        self.assertEqual(metadata['excluded_conflicting_images'], 3)
        excluded = read_rows(self.audit / 'conflicting_label_duplicates.csv')
        self.assertEqual({row['label'] for row in excluded}, {'0', '1'})
        shutil.rmtree(self.splits)
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            prepare_split(self.data, self.audit, self.splits, expected_classes=3,
                          conflicting_label_policy='error')

    def test_single_group_class_is_train_only(self):
        shutil.rmtree(self.data / '2')
        (self.data / '2').mkdir()
        Image.new('RGB', (20, 24), (255, 255, 255)).save(self.data / '2/only.png')
        self.run_audit()
        metadata = self.split()
        self.assertEqual(metadata['train_only_classes'], ['2'])
        self.assertIn('2', {row['label'] for row in read_rows(self.splits / 'train.csv')})
        self.assertNotIn('2', {row['label'] for row in read_rows(self.splits / 'val.csv')})

    def test_large_duplicate_group_does_not_dominate_validation(self):
        for i in range(40):
            shutil.copyfile(self.data / '0/0.png', self.data / f'0/copy-{i}.png')
        self.run_audit()
        self.split()
        val = read_rows(self.splits / 'val.csv')
        self.assertFalse(any(r['path'].startswith('0/copy-') for r in val))

    def test_augmented_family_stays_out_of_validation_and_cannot_leak(self):
        Image.new('RGB', (20, 24), (1, 2, 3)).save(self.data / '0/1__aug_d3.png')
        Image.new('RGB', (20, 24), (4, 5, 6)).save(self.data / '0/1__erode_k3_it1__aug_shift5.png')
        self.run_audit()
        metadata = self.split()
        train, val = read_rows(self.splits / 'train.csv'), read_rows(self.splits / 'val.csv')
        self.assertFalse({r['source_id'] for r in train} & {r['source_id'] for r in val})
        self.assertFalse(any('__aug_' in r['path'] for r in val))
        family = source_id_for_path('0/1.png')
        self.assertEqual(family, source_id_for_path('0/1__aug_d3.png'))
        if family in {r['source_id'] for r in val}:
            self.assertFalse(any(r['source_id'] == family for r in train))
            self.assertGreaterEqual(metadata['held_out_derived_images'], 2)
        else:
            self.assertEqual(sum(r['source_id'] == family for r in train), 3)


if __name__ == '__main__':
    unittest.main()

"""Team jobs cover the catalog once; notebooks record failed runs honestly."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import nbformat

ROOT = Path(__file__).resolve().parents[1]


class TeamNotebookTest(unittest.TestCase):
    def setUp(self):
        self.template = nbformat.read(ROOT / 'notebooks/03-team-model-template.ipynb', as_version=4)

    def cell(self, tag):
        return next(c.source for c in self.template.cells if tag in c.metadata.get('tags', []))

    def test_job_catalog_and_round_allocation(self):
        plan = json.loads((ROOT / 'configs/experiments/team_family_assignments.json').read_text())
        lab = nbformat.read(ROOT / 'notebooks/02-model-family-search-lab.ipynb', as_version=4)
        candidates = next(c.source for c in lab.cells if 'candidates' in c.metadata.get('tags', []))
        original = ast.literal_eval(ast.parse(candidates).body[0].value)
        self.assertEqual([(c['family'], c['model']) for c in plan['candidates']], [r[:2] for r in original])
        self.assertEqual(len(plan['candidates']), 40)

        member_paths = sorted((ROOT / 'notebooks/team').glob('member-*.ipynb'))
        self.assertEqual(len(member_paths), 7)
        for member_id, path in enumerate(member_paths, start=1):
            raw = json.loads(path.read_text())
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            for raw_cell, cell in zip(raw['cells'], notebook.cells):
                self.assertIn('id', raw_cell)
                if cell.cell_type == 'code':
                    compile(cell.source, str(path), 'exec')
                    self.assertFalse(cell.outputs)
            member_ns = {}
            exec(next(c.source for c in notebook.cells if 'settings' in c.metadata.get('tags', [])), member_ns)
            self.assertEqual(member_ns['MEMBER_ID'], member_id)
            self.assertEqual(member_ns['ROUND'], 1)
            self.assertIsNone(member_ns['JOB_ID'])

        ns = {}
        exec(self.cell('settings'), ns)
        with tempfile.TemporaryDirectory() as directory:
            ns.update(OWNER='tester', REPO_PATH=str(ROOT), OUTPUT_ROOT=directory, DEVICE='cpu')
            imports = self.cell('imports')
            for size in (6, 7):
                jobs = []
                for round_number in range(1, 8):
                    for member_id in range(1, size + 1):
                        expected = (round_number - 1) * size + member_id
                        if expected > 41:
                            continue
                        run_ns = {**ns, 'TEAM_SIZE': size, 'MEMBER_ID': member_id,
                                  'ROUND': round_number, 'JOB_ID': None}
                        exec(imports, run_ns)
                        jobs.append(run_ns['assigned_job_id'])
                self.assertEqual(jobs, list(range(1, 42)))

            manual_ns = {**ns, 'TEAM_SIZE': 7, 'MEMBER_ID': 1, 'ROUND': 1, 'JOB_ID': 41}
            exec(imports, manual_ns)
            manual_ns['display'] = lambda *args: None
            exec(self.cell('assignments'), manual_ns)
            self.assertEqual(manual_ns['assigned_job_id'], 41)
            self.assertEqual(len(manual_ns['selected']), 1)
            self.assertEqual(len(manual_ns['task_table']), 41)
            self.assertFalse(manual_ns['task_table'].family.duplicated().any())

    def test_failed_job_writes_handoff(self):
        ns = {}
        exec(self.cell('settings'), ns)
        with tempfile.TemporaryDirectory() as directory:
            ns.update(OWNER='tester', REPO_PATH=str(ROOT), OUTPUT_ROOT=directory, DEVICE='cpu')
            exec(self.cell('imports'), ns)
            ns['display'] = lambda *args: None
            exec(self.cell('assignments'), ns)
            ns['split_meta'] = {'split_hash': 'test-only'}
            exec(self.cell('protocol'), ns)
            # Exercise the actual job, without downloads, preflight compute, or training.
            ns.update(mapping={'a': 0, 'b': 1}, data_dir=Path(directory), split_dir=Path(directory))
            ns['fit'] = Mock(side_effect=RuntimeError('test failure'))
            tree = ast.parse(self.cell('screening'))
            tree.body = [node for node in tree.body if not isinstance(node, ast.FunctionDef)]
            ns['preflight'] = Mock(return_value=10)
            exec(compile(tree, 'screening', 'exec'), ns)
            exec(self.cell('handoff'), ns)
            handoff = json.loads((ns['experiment_dir'] / 'handoff.json').read_text())
            self.assertFalse(handoff['complete'])
            self.assertEqual(handoff['job_id'], 1)
            self.assertEqual(len(handoff['failed']), 1)
            self.assertEqual(len(handoff['completed_families']), 0)
            self.assertEqual(ns['fit'].call_count, 1)
            self.assertEqual([s['status'] for s in ns['statuses']], ['failed'])
            self.assertTrue(ns['screening'].empty)


if __name__ == '__main__':
    unittest.main()

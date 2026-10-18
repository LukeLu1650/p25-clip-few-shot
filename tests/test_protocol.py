import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from p25.data import digest, sample, stratify, validate_manifest


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"path": f"c{c}/{i}.jpg", "label": c} for c in range(3) for i in range(40)]

    def test_partition_fixed_and_shots_nested(self):
        pools = stratify(self.rows)
        self.assertEqual(pools, stratify(list(reversed(self.rows))))
        a = sample(pools['train'], 1, 1)
        b = sample(pools['train'], 16, 1)
        self.assertTrue({r['path'] for r in a} <= {r['path'] for r in b})
        self.assertNotEqual(b, sample(pools['train'], 16, 2))
        m = dict(classes=['a', 'b', 'c'], shots=16, val_shots=4,
                 train=b, val=sample(pools['val'], 4, 0), test=pools['test'])
        validate_manifest(m)
        m['val'][0] = m['train'][0]
        with self.assertRaises(ValueError):
            validate_manifest(m)

    def test_no_silent_replacement_sampling(self):
        with self.assertRaises(ValueError):
            sample(self.rows, 41, 1)

    def test_summary_requires_distinct_seeds(self):
        spec = importlib.util.spec_from_file_location('summary', Path(__file__).resolve().parents[1] / 'scripts/summarize.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            for seed, acc in [(1, 70), (2, 80), (3, 90)]:
                p = Path(folder) / str(seed) / 'metrics.json'
                p.parent.mkdir()
                obj = dict(status='complete', config=dict(dataset='toy', method='coop', shots=1, seed=seed),
                           pool_sha256='pool', accuracy={'0': acc}, trainable_parameters=8,
                           timing={'cold_adaptation_seconds': 1})
                p.write_text(json.dumps(obj))
            rows = module.summarize(folder)
            self.assertEqual(rows[0]['mean_accuracy'], 80)
            self.assertEqual(rows[0]['sample_std'], 10)
            self.assertTrue(rows[0]['sufficient_seeds'])
            dup = Path(folder) / 'duplicate'
            dup.mkdir()
            (dup / 'metrics.json').write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError, 'Duplicate seed'):
                module.summarize(folder)


if __name__ == '__main__':
    unittest.main()

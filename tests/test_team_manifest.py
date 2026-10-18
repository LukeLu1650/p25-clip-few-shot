import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from p25.data import import_team_eurosat


@unittest.skipUnless(os.environ.get('P25_TEST_DATA'), 'Set P25_TEST_DATA to test actual teammate manifests')
class TeamManifestTests(unittest.TestCase):
    def test_exact_partition_sizes_and_nested_sampling(self):
        root = Path(os.environ['P25_TEST_DATA'])
        small = import_team_eurosat(root, root / 'manifests', 1, 1)
        large = import_team_eurosat(root, root / 'manifests', 16, 1)
        other = import_team_eurosat(root, root / 'manifests', 16, 2)
        self.assertEqual(len(small['train']), 10)
        self.assertEqual(len(large['train']), 160)
        self.assertEqual(len(small['val']), 2700)
        self.assertEqual(len(small['test']), 8100)
        self.assertEqual(small['split_seed'], 2026)
        self.assertEqual(small['val'], other['val'])
        self.assertEqual(small['test'], other['test'])
        self.assertTrue({r['path'] for r in small['train']} <= {r['path'] for r in large['train']})
        self.assertNotEqual(large['train'], other['train'])


if __name__ == '__main__':
    unittest.main()

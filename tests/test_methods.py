import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from torch.nn import functional as F
from p25.methods import CoOp, text_weights, tip_logits, train_coop, tune_tip


class TipTests(unittest.TestCase):
    def test_manual_cache_vote_and_alpha_zero(self):
        x = torch.eye(2)
        values = torch.eye(2)
        expected = 100 * x + 2 * torch.tensor([[1., 0.36787944], [0.36787944, 1.]])
        self.assertTrue(torch.allclose(tip_logits(x, x, x, values, 2, 1), expected))
        self.assertTrue(torch.equal(tip_logits(x, x, x, values, 0, 1), 100 * x))
        selected = tune_tip(x, torch.tensor([0, 1]), x, x, values, [0, 1], [1, 2])
        self.assertEqual(selected['val_accuracy'], 100)
        self.assertEqual(selected['alpha'], 0)


@unittest.skipUnless(os.environ.get('P25_TEST_CLIP_CACHE'), 'Set P25_TEST_CLIP_CACHE for real CLIP integration')
class RealCLIPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import clip
        cls.model, _ = clip.load('ViT-B/32', device='cuda' if torch.cuda.is_available() else 'cpu',
                                  jit=False, download_root=os.environ['P25_TEST_CLIP_CACHE'])
        cls.model.float().eval().requires_grad_(False)

    def test_word_initialization_matches_clip_and_context_receives_gradients(self):
        classes = ['forest', 'river']
        prompt = CoOp(self.model, classes, n_ctx=4, ctx_init='a photo of a')
        expected = text_weights(self.model, classes, 'a photo of a {}.')
        self.assertTrue(torch.allclose(prompt.text_features(), expected, atol=2e-5))
        device = next(self.model.parameters()).device
        features = F.normalize(torch.randn(4, 512, device=device), dim=-1)
        labels = torch.tensor([0, 1, 0, 1], device=device)
        before = prompt.context.detach().clone()
        result = train_coop(prompt, (features, labels), (features, labels), 2, 2, .002, 1)
        self.assertFalse(torch.equal(before, prompt.context))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertEqual(len(result['history']), 2)
        self.assertEqual(sum(p.numel() for p in prompt.parameters() if p.requires_grad), 4 * 512)

    def test_initialization_length_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'BPE tokens'):
            CoOp(self.model, ['forest'], n_ctx=16, ctx_init='a photo of a')


if __name__ == '__main__':
    unittest.main()

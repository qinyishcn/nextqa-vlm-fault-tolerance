import unittest

try:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from prefix_cache import generate_from_prefix, last_position_logits, warm_prefix
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Requires the separate Torch/Transformers inference environment')
class GenerationParityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261012)
        config = Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=512, bos_token_id=0, eos_token_id=None, pad_token_id=0,
            attention_dropout=0.0, _attn_implementation='eager')
        self.model = Qwen2ForCausalLM(config).eval()

    def test_shared_cache_preserves_two_distinct_suffix_generations(self):
        prefix = torch.tensor([[1, 2, 3, 4, 5, 6]])
        cache = warm_prefix(self.model, self.model.model.embed_tokens(prefix), chunk_size=2)
        saved = tuple(tuple(value.clone() for value in layer) for layer in cache)
        for suffix in [torch.tensor([[7, 8, 9]]), torch.tensor([[10, 11, 12, 13]])]:
            complete = torch.cat([prefix, suffix], dim=1)
            reference = self.model.generate(complete, do_sample=False, repetition_penalty=1.0,
                max_new_tokens=8, attention_mask=torch.ones_like(complete))[:, complete.shape[1]:]
            received = generate_from_prefix(self.model, suffix, cache)
            self.assertTrue(torch.equal(reference, received))
        self.assertTrue(all(torch.equal(a, b) for original, after in zip(saved, cache)
                            for a, b in zip(original, after)))

    def test_last_position_head_preserves_logits_and_generated_tokens(self):
        inputs = torch.tensor([[1, 2, 3, 4, 5, 6]])
        with torch.inference_mode():
            reference = self.model(inputs).logits[:, -1:, :]
            full = self.model.generate(inputs, do_sample=False, repetition_penalty=1.0,
                attention_mask=torch.ones_like(inputs), max_new_tokens=8)
            with last_position_logits(self.model):
                received = self.model(inputs).logits
                slim = self.model.generate(inputs, do_sample=False, repetition_penalty=1.0,
                    attention_mask=torch.ones_like(inputs), max_new_tokens=8)
        torch.testing.assert_close(reference, received)
        self.assertTrue(torch.equal(full, slim))

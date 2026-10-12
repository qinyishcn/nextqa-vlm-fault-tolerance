"""Use Hugging Face generation with an immutable shared causal prefix cache."""
from contextlib import contextmanager

import torch
from transformers import Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache


@contextmanager
def last_position_logits(model):
    """Avoid allocating [video tokens, vocabulary] logits during inference.

    Greedy generation consumes only the final position's logits. The hidden
    states, attention and KV cache are unchanged; training/labels are unsupported.
    """
    handle = model.lm_head.register_forward_pre_hook(lambda module, args: (args[0][:, -1:, :],))
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def warm_prefix(model, embeddings, chunk_size=1024):
    backbone = model.get_model() if hasattr(model, 'get_model') else model.model
    if chunk_size < 1:
        raise ValueError('Prefill chunk size must be positive')
    cache = DynamicCache()
    for start in range(0, embeddings.shape[1], chunk_size):
        end = min(start + chunk_size, embeddings.shape[1])
        mask = torch.ones((1, end), dtype=torch.long, device=embeddings.device)
        result = backbone(inputs_embeds=embeddings[:, start:end], past_key_values=cache,
            attention_mask=mask, use_cache=True, output_hidden_states=False,
            output_attentions=False, return_dict=True)
        cache = result.past_key_values
    if hasattr(cache, 'to_legacy_cache'):
        cache = cache.to_legacy_cache()
    if cache[0][0].shape[-2] != embeddings.shape[1]:
        raise ValueError('Shared prefix cache length mismatch')
    return cache


@torch.inference_mode()
def generate_from_prefix(model, suffix_ids, cache, max_new_tokens=8):
    """Cached positions use dummy IDs; only uncached suffix IDs enter the model.

    Repetition penalty is explicitly 1, so dummy cached IDs cannot influence a
    logits processor. This calls HF generation, retaining EOS and cache handling.
    """
    prefix_length = cache[0][0].shape[-2]
    dummy = torch.full((1, prefix_length), model.generation_config.pad_token_id,
                       dtype=torch.long, device=suffix_ids.device)
    inputs = torch.cat([dummy, suffix_ids], dim=1)
    mask = torch.ones_like(inputs)
    outputs = Qwen2ForCausalLM.generate(model, inputs=inputs, attention_mask=mask,
        past_key_values=cache, use_cache=True, do_sample=False, num_beams=1,
        repetition_penalty=1.0, max_new_tokens=max_new_tokens)
    if cache[0][0].shape[-2] != prefix_length:
        raise ValueError('Generation mutated the shared prefix cache')
    return outputs[:, inputs.shape[1]:]

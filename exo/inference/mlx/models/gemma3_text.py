from dataclasses import dataclass, field
import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.gemma3_text import (
    ModelArgs as Gemma3Args,
    TransformerBlock,
    RMSNorm,
    KVCache, RotatingKVCache
)
from ...shard import Shard
from .base import IdentityBlock


@dataclass
class ModelArgs(Gemma3Args):
    shard: Shard = field(default_factory=lambda: Shard("", 0, 0, 0))

    def __post_init__(self):
        if isinstance(self.shard, Shard):
            return
        if not isinstance(self.shard, dict):
            raise TypeError(f"Expected shard to be a Shard or dict, got {type(self.shard)}")
        self.shard = Shard(**self.shard)


class ShardedGemma3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0

        # Embeddings only on first/last shard (so last shard can tie if needed)
        if args.shard.is_first_layer() or args.shard.is_last_layer():
            self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

        # Place real blocks only for our shard; Identity elsewhere
        self.layers = [
            (TransformerBlock(args=args, layer_idx=i)
             if args.shard.start_layer <= i <= args.shard.end_layer
             else IdentityBlock())
            for i in range(args.num_hidden_layers)
        ]

        # Final norm only on last shard
        if args.shard.is_last_layer():
            self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, mask=None, cache=None, input_embeddings=None):
        # First shard does token embed; others receive hidden states in `inputs`
        if self.args.shard.is_first_layer():
            h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
            # IMPORTANT: Gemma-3 uses bf16 scale cast
            scale = mx.array(self.args.hidden_size ** 0.5, mx.bfloat16).astype(h.dtype)
            h = h * scale
        else:
            h = inputs  # already hidden states

        if cache is None:
            cache = [None] * len(self.layers)

        # Recreate Gemma-3 mask logic (sliding vs global layers)
        if mask is None:
            j = self.args.sliding_window_pattern
            full_mask = create_attention_mask(h, cache[j - 1 : j])   # for global layers
            sliding_window_mask = create_attention_mask(h, cache)     # for sliding layers

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            # IdentityBlock just returns h; but we still need the right mask for real layers
            is_global = (i % self.args.sliding_window_pattern) == (self.args.sliding_window_pattern - 1)

            local_mask = mask
            if mask is None and is_global:
                local_mask = full_mask
            elif mask is None:
                local_mask = sliding_window_mask

            h = layer(h, local_mask, c)

        if self.args.shard.is_last_layer():
            h = self.norm(h)

        return h
    
    def make_cache(self):
        caches = []
        pat = self.args.sliding_window_pattern
        for i in range(self.args.num_hidden_layers):
            if (i % pat) == (pat - 1):
                caches.append(KVCache())  # global layer
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window, keep=0))
        return caches


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = ShardedGemma3Model(args)

        if args.shard.is_last_layer():
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
            self.tie_word_embeddings = False

    def __call__(self, inputs, cache=None, mask=None, input_embeddings=None):
        out = self.model(inputs, mask, cache, input_embeddings)
        if self.args.shard.is_last_layer():
            if getattr(self, "tie_word_embeddings", False):
                out = self.model.embed_tokens.as_linear(out)
            else:
                out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        if "lm_head.weight" not in weights:
            self.tie_word_embeddings = True
            if hasattr(self, "lm_head"):
                self.pop("lm_head")

        shard_state = {}
        for key, value in weights.items():
            if key.startswith("model.layers."):
                layer_num = int(key.split(".")[2])
                if self.args.shard.start_layer <= layer_num <= self.args.shard.end_layer:
                    shard_state[key] = value
            elif (self.args.shard.is_first_layer() or self.args.shard.is_last_layer()) and key.startswith("model.embed_tokens"):
                shard_state[key] = value
            elif self.args.shard.is_last_layer() and key.startswith("model.norm"):
                shard_state[key] = value
            elif self.args.shard.is_last_layer() and key.startswith("lm_head"):
                shard_state[key] = value
        return shard_state
    
    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    def make_cache(self):
        return self.model.make_cache()
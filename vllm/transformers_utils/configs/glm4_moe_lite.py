# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The ZhipuAI Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration class for GLM-4.7-Flash (glm4_moe_lite) models.

Defined here because transformers does not yet register the
``glm4_moe_lite`` model type, so ``AutoConfig.from_pretrained`` would
fail when loading such a checkpoint.  vLLM's ``_CONFIG_REGISTRY`` maps
the ``model_type`` string to this class before falling back to the
transformers registry.
"""

from transformers import PretrainedConfig


class Glm4MoeLiteConfig(PretrainedConfig):
    r"""
    Configuration for GLM-4 MoE Lite (glm4_moe_lite) models.

    Structurally identical to ``Glm4MoeConfig`` but with
    ``model_type = "glm4_moe_lite"``, which lets vLLM load checkpoints
    whose ``config.json`` carries that model type without requiring a
    transformers version that already knows about it.

    All parameters are optional; their defaults match a typical
    GLM-4.7-Flash configuration but the actual values are always
    read from the checkpoint's ``config.json``.
    """

    model_type = "glm4_moe_lite"

    def __init__(
        self,
        vocab_size: int = 151552,
        hidden_size: int = 2048,
        intermediate_size: int = 5472,
        moe_intermediate_size: int = 1536,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-6,
        # MoE routing
        n_routed_experts: int = 64,
        n_shared_experts: int | None = 1,
        num_experts_per_tok: int = 6,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        first_k_dense_replace: int = 1,
        moe_layer_freq: int = 1,
        n_group: int = 1,
        topk_group: int = 1,
        # MLA (multi-head latent attention) fields
        qk_nope_head_dim: int = 64,
        qk_rope_head_dim: int = 32,
        v_head_dim: int = 128,
        kv_lora_rank: int | None = None,
        q_lora_rank: int | None = None,
        # speculative / MTP
        num_nextn_predict_layers: int = 0,
        # misc
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_key_value_heads
            if num_key_value_heads is not None
            else num_attention_heads
        )
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        # MoE
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.n_group = n_group
        self.topk_group = topk_group
        # MLA
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        # speculative
        self.num_nextn_predict_layers = num_nextn_predict_layers

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["Glm4MoeLiteConfig"]

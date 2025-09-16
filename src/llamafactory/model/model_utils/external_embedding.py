# Copyright 2025 the LlamaFactory team.
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

from types import MethodType
from typing import Optional

import torch
from torch import nn

from ...extras import logging


logger = logging.get_logger(__name__)


class ExternalEmbeddingProjector(nn.Module):
    """Project external embeddings to the language model hidden space."""

    def __init__(self, input_dim: int, hidden_size: int, use_bias: bool = True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.projection = nn.Linear(input_dim, hidden_size, bias=use_bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.numel() == 0:
            return embeddings.new_zeros(embeddings.size(0), embeddings.size(1), self.hidden_size)

        original_shape = embeddings.shape
        projected = self.projection(embeddings.view(-1, original_shape[-1]))
        return projected.view(original_shape[0], original_shape[1], self.hidden_size)


def _infer_hidden_size(model: "torch.nn.Module") -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Cannot infer hidden size without model config.")

    candidates = [config]
    if getattr(config, "text_config", None) is not None:
        candidates.append(config.text_config)

    attr_names = ["hidden_size", "hidden_dim", "n_embd", "d_model", "model_dim", "dim"]
    for candidate in candidates:
        for attr in attr_names:
            value = getattr(candidate, attr, None)
            if isinstance(value, int) and value > 0:
                return value

    raise ValueError("Cannot infer model hidden size for external embedding projector.")


def _get_weight_dtype(model: "torch.nn.Module") -> torch.dtype:
    embedding = getattr(model, "get_input_embeddings", None)
    if callable(embedding):
        weight = embedding().weight
        if weight is not None:
            return weight.dtype

    try:
        parameter = next(model.parameters())
        return parameter.dtype
    except StopIteration:  # pragma: no cover - safeguard
        return torch.get_default_dtype()


def attach_external_embedding_module(
    model: "torch.nn.Module",
    external_dim: Optional[int],
    use_bias: bool,
    position: str,
) -> Optional[ExternalEmbeddingProjector]:
    """Attach the projection module and patch forward for external embeddings."""

    if external_dim is None:
        return None

    if external_dim <= 0:
        raise ValueError("`external_embedding_dim` must be a positive integer.")

    if hasattr(model, "external_embedding_projector"):
        projector = getattr(model, "external_embedding_projector")
        if not isinstance(projector, ExternalEmbeddingProjector):
            raise TypeError("`external_embedding_projector` already exists with an unsupported type.")

        if projector.input_dim != external_dim or projector.hidden_size != _infer_hidden_size(model):
            logger.warning_rank0("Reinitializing external embedding projector due to configuration changes.")
            projector = ExternalEmbeddingProjector(external_dim, _infer_hidden_size(model), use_bias)
            setattr(model, "external_embedding_projector", projector)
    else:
        projector = ExternalEmbeddingProjector(external_dim, _infer_hidden_size(model), use_bias)
        setattr(model, "external_embedding_projector", projector)

    try:
        first_param = next(model.parameters())
    except StopIteration:  # pragma: no cover - safeguard for empty modules
        first_param = None

    device = first_param.device if first_param is not None else torch.device("cpu")
    projector.to(device=device, dtype=_get_weight_dtype(model))

    config = getattr(model, "config", None)
    if config is not None:
        setattr(config, "external_embedding_dim", external_dim)
        setattr(config, "external_embedding_use_bias", use_bias)
        setattr(config, "external_embedding_position", position)

    _patch_model_forward(model)
    return projector


def _patch_model_forward(model: "torch.nn.Module") -> None:
    if getattr(model, "_external_embedding_forward_patched", False):
        return

    raw_forward = model.forward

    def forward(
        self,
        *args,
        external_embeddings: Optional[torch.Tensor] = None,
        external_attention_mask: Optional[torch.Tensor] = None,
        external_token_count: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if external_embeddings is not None and external_attention_mask is not None:
            projector: ExternalEmbeddingProjector = getattr(self, "external_embedding_projector")
            if projector is None:
                raise ValueError("External embeddings are provided but the projector is missing.")

            input_ids = kwargs.get("input_ids")
            inputs_embeds = kwargs.get("inputs_embeds")

            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("Either `input_ids` or `inputs_embeds` must be provided.")

                embed_layer = self.get_input_embeddings()
                inputs_embeds = embed_layer(input_ids)
                kwargs["input_ids"] = None

            # Avoid in-place modifications on a view that would break autograd.
            inputs_embeds = inputs_embeds.clone()
            kwargs["inputs_embeds"] = inputs_embeds

            counts = (
                external_token_count.to(inputs_embeds.device)
                if external_token_count is not None
                else external_attention_mask.sum(dim=-1).to(inputs_embeds.device)
            )

            max_tokens = external_embeddings.size(1)
            if max_tokens > 0:
                projected = projector(external_embeddings.to(inputs_embeds.device))
                projected = projected.to(inputs_embeds.dtype)

                prefix = inputs_embeds[:, :max_tokens, :]
                if prefix.size(1) < max_tokens:
                    raise ValueError("Input ids must reserve placeholder positions for external embeddings.")

                mask = (
                    torch.arange(max_tokens, device=prefix.device).unsqueeze(0) < counts.unsqueeze(1)
                ).unsqueeze(-1)
                updated_prefix = torch.where(mask, projected, prefix)
                inputs_embeds = torch.cat((updated_prefix, inputs_embeds[:, max_tokens:, :]), dim=1)
                kwargs["inputs_embeds"] = inputs_embeds

        kwargs.pop("external_embeddings", None)
        kwargs.pop("external_attention_mask", None)
        kwargs.pop("external_token_count", None)

        return raw_forward(*args, **kwargs)

    model.forward = MethodType(forward, model)
    setattr(model, "_external_embedding_forward_patched", True)


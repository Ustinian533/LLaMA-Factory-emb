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

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch

from ..extras import logging


logger = logging.get_logger(__name__)


class ExternalEmbeddingLibrary:
    """Load and resolve embeddings stored in a torch serialized dictionary."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Embedding library file not found: {self.path}")

        raw = torch.load(self.path, map_location="cpu")
        if not isinstance(raw, dict):
            raise TypeError("Embedding library must be a dictionary mapping id -> tensor or list.")

        self._embeddings: Dict[str, torch.Tensor] = {}
        dims: set[int] = set()
        for key, value in raw.items():
            tensor = torch.as_tensor(value, dtype=torch.float32)
            if tensor.dim() != 1:
                raise ValueError(
                    "Each embedding must be one-dimensional (representing a single token vector)."
                )

            embedding_id = str(key)
            self._embeddings[embedding_id] = tensor.clone().detach()
            dims.add(tensor.size(0))

        if len(self._embeddings) == 0:
            raise ValueError("Embedding library is empty.")

        if len(dims) != 1:
            raise ValueError("Embeddings in the library must all share the same dimensionality.")

        self.embedding_dim = dims.pop()
        logger.info_rank0(
            "Loaded %d external embeddings from %s (dim=%d).",
            len(self._embeddings),
            self.path,
            self.embedding_dim,
        )

    def resolve(self, embedding_id: str) -> torch.Tensor:
        try:
            return self._embeddings[embedding_id].clone().detach()
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise KeyError(f"Embedding id {embedding_id!r} not found in {self.path}.") from exc

    def batch_resolve(self, embedding_ids: Sequence[str]) -> torch.Tensor:
        if len(embedding_ids) == 0:
            return torch.zeros((0, self.embedding_dim), dtype=torch.float32)

        resolved: List[torch.Tensor] = [self.resolve(str(idx)) for idx in embedding_ids]
        return torch.stack(resolved, dim=0)

    def __contains__(self, embedding_id: str) -> bool:
        return embedding_id in self._embeddings

    def keys(self) -> Iterable[str]:  # pragma: no cover - helper for debugging
        return self._embeddings.keys()

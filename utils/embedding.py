# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

from typing import Optional

import torch
from jaxtyping import Float, Int, Num


def patchify_id_mask(
    id_mask: torch.Tensor, patch_size: int
) -> torch.Tensor:
    patched_id_mask = id_mask.unfold(dimension=-1, size=patch_size, step=patch_size)
    patched_id_mask_min = patched_id_mask.min(-1).values
    patched_id_mask_max = patched_id_mask.max(-1).values
    assert torch.eq(patched_id_mask_min, patched_id_mask_max).all(), "Patches cannot span multiple datasets"
    return patched_id_mask_min


class PatchEmbedding(torch.nn.Module):
    """
    Multivariate time series patch embedding.
    Patchifies each variate separately.
    """

    def __init__(self, patch_size: int, stride: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.stride = stride
        self.projection = torch.nn.Linear(self.patch_size, self.embed_dim)

    def _patchify(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        return x.unfold(dimension=-1, size=self.patch_size, step=self.stride)

    def forward(
        self,
        x: torch.Tensor,
        id_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        assert (
            x.shape[-1] % self.patch_size == 0
        ), f"Series length ({x.shape=}) must be divisible by ({self.patch_size=})"
        x_patched: torch.Tensor = self._patchify(x)
        id_mask_patched: torch.Tensor = self._patchify(id_mask)

        assert torch.eq(
            id_mask_patched.min(-1).values, id_mask_patched.max(-1).values
        ).all(), "Patches cannot span multiple datasets"

        return (
            self.projection(x_patched),
            id_mask_patched.min(-1).values,
        )

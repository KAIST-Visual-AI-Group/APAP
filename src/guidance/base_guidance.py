"""
base_guidance.py

The base class for diffusion guidances.
"""

from abc import ABC, abstractmethod


class Guidance(ABC):
    """The base class of guidances."""

    def __init__(self, *args, **kwargs) -> None:
        """Constructor"""
        pass  # do nothing

    @abstractmethod
    def _build_model(self) -> None:
        """Builds the model"""

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Forward call"""

    @abstractmethod
    def compute_img_grad(
        self,
        *args,
        **kwargs,
    ):
        """Computes SDS loss"""

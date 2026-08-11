"""Host-to-device transfer abstractions for expert offload."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class H2DCopyTask:
    """One byte-preserving host-to-device copy.

    ``source`` and ``destination`` intentionally accept both tensors and
    untyped-storage slices. Expert weights use storage slices to preserve their
    packed/NZ byte layout, while quantization attributes use shaped tensors.
    ``nbytes`` is carried explicitly so pointer-based backends can consume the
    same task without inferring sizes from tensor metadata.
    """

    source: Any
    destination: Any
    nbytes: int
    name: str = ""
    non_blocking: bool = True

    def __post_init__(self) -> None:
        if self.nbytes <= 0:
            raise ValueError(f"H2D copy size must be positive; got {self.nbytes}")


class ExpertH2DTransport(Protocol):
    """Backend contract for a batch of expert H2D copies."""

    def copy_batch(self, tasks: list[H2DCopyTask]) -> None:
        """Enqueue all tasks on the caller-selected device stream."""
        ...


class TorchCopyH2DTransport:
    """Existing expert-offload H2D behavior implemented with ``copy_``."""

    def copy_batch(self, tasks: list[H2DCopyTask]) -> None:
        for task in tasks:
            task.destination.copy_(task.source,
                                   non_blocking=task.non_blocking)

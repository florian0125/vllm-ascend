"""Locate the byte-copy library built for this source checkout."""
from pathlib import Path


def get_default_copy_library() -> str:
    library = Path(__file__).parent / "lib" / "libnpu_expert_copy_dispatch.so"
    if not library.is_file():
        raise FileNotFoundError(
            "NPU expert copy library is not built. Source CANN and run "
            "bash csrc/expert_transfer_copy/build.sh "
            '"$PWD/build/expert_transfer_copy", or set transfer_copy_library.'
        )
    return str(library)

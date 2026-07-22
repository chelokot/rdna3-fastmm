from pathlib import Path
import sys


repository_root = Path(__file__).resolve().parent
source_root = repository_root / "src"
loaded_package = sys.modules.get("rdna3_fastmm")

if loaded_package is not None:
    loaded_file = getattr(loaded_package, "__file__", None)
    if loaded_file is None or source_root not in Path(loaded_file).resolve().parents:
        raise RuntimeError(
            "A different rdna3_fastmm package is already loaded; restart ComfyUI "
            "after removing the conflicting installation"
        )

sys.path.insert(0, str(source_root))
try:
    from rdna3_fastmm.comfyui import (
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
    )
finally:
    sys.path.pop(0)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

from __future__ import annotations

import types
import sys
from pathlib import Path


def get_clip_module():
    try:
        import clip as clip_module

        return clip_module
    except ModuleNotFoundError:
        fallback_root = Path(__file__).resolve().parents[1] / "_tmp_tda_repo"
        if fallback_root.exists() and str(fallback_root) not in sys.path:
            sys.path.insert(0, str(fallback_root))
            try:
                import ftfy  # noqa: F401
            except ModuleNotFoundError:
                shim = types.ModuleType("ftfy")
                shim.fix_text = lambda text: text
                sys.modules["ftfy"] = shim
            import clip as clip_module

            return clip_module
        raise

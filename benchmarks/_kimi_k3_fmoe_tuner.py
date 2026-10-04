"""Importable facade for the FMoE tuner bundled with the Kimi-K3 image.

The AITER multiprocessing tuner pickles its task functions by module name, so
the image source must execute in a module that spawned workers can import.
"""

from aiter.jit.core import AITER_CSRC_DIR


_SOURCE = f"{AITER_CSRC_DIR}/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py"
with open(_SOURCE, "rb") as _stream:
    _source_text = _stream.read().decode("utf-8")

# AMD's production K3 table overwhelmingly selects the stage-2 reduction
# kernel, but this image revision's tuner accidentally filters every non-atomic
# stage-2 candidate.  Restore those registered reduction candidates.
_atomic_only = '''                if kparams.get("mode", "atomic") != "atomic":
                    continue
'''
if _atomic_only not in _source_text:
    raise RuntimeError("The image FMoE tuner's atomic-only filter changed")
_source_text = _source_text.replace(_atomic_only, "", 1)

exec(compile(_source_text, _SOURCE, "exec"), globals(), globals())

"""Compatibility launcher for the staff ``setup_data.py --prefix-banks`` step.

``scripts/setup_data.py`` (course dataset repo, not ours to edit) calls ``model.generate(..., generator=gen)`` with a seeded
``torch.Generator``. ``transformers`` 4.57.x does not forward a ``generator`` kwarg to sampling (``generate`` raises
``ValueError: The following model_kwargs are not used by the model: ['generator']``); sampling then draws from the *global* RNG of
the device. This launcher keeps the staff script byte-for-byte unchanged and makes the intended semantics work: while ``generate``
runs, the device's global RNG state is swapped for the generator's state (same Philox stream a ``torch.Generator`` would use),
and the advanced state is written back to the generator afterwards. Per-prompt seeds (6304 + 10007 j + offset) therefore keep
producing deterministic, prompt-independent banks. No-op when ``generator`` is not passed.

Usage (cwd = dataset repo root):  python -m pa4t3.data_compat --prefix-banks
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch


def _with_generator(orig):
    def generate(self, *args, generator=None, **kwargs):
        if generator is None:
            return orig(self, *args, **kwargs)
        dev = torch.device(generator.device)
        if dev.type == "cuda":
            saved = torch.cuda.get_rng_state(dev)
            torch.cuda.set_rng_state(generator.get_state(), dev)
            get_state, set_state = (lambda: torch.cuda.get_rng_state(dev)), (lambda s: torch.cuda.set_rng_state(s, dev))
        else:
            saved = torch.get_rng_state()
            torch.set_rng_state(generator.get_state())
            get_state, set_state = torch.get_rng_state, torch.set_rng_state
        try:
            return orig(self, *args, **kwargs)
        finally:
            generator.set_state(get_state())
            set_state(saved)
    generate._pa4_generator_shim, generate._pa4_orig = True, orig
    return generate


def install():
    from transformers import GenerationMixin
    if not getattr(GenerationMixin.generate, "_pa4_generator_shim", False):
        GenerationMixin.generate = _with_generator(GenerationMixin.generate)


def main():
    install()
    script = Path("scripts/setup_data.py")
    if not script.exists():
        raise SystemExit("run from the dataset repo root (scripts/setup_data.py not found)")
    sys.argv = [str(script)] + sys.argv[1:]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()

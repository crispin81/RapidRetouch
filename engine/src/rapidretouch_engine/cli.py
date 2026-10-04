"""Command line: `rapidretouch-engine serve` for the app, or run a tool on one file."""

from __future__ import annotations

import os

# Must be set before torch loads. Lets PyTorch grow and shrink GPU memory in
# place, so memory freed after a run can actually be reused, and returned to
# other apps sharing the card, instead of stranding in fragments.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# On a Mac's GPU, run any operation it lacks on the CPU instead of failing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
from dataclasses import fields

from .tools import backdrop_smooth


def prepare() -> None:
    """Run by the app's first-launch setup, after installing the engine: the
    default models are downloaded and every library is loaded once, so that
    none of it happens later in the middle of editing (on a Mac, loading a
    library the first time also has it security-scanned, which is slow).
    Progress goes to stdout as {"message", "fraction"} lines; a model that
    can't be downloaded now is left for when a tool needs it."""
    import importlib

    from .registry import Registry, prefetch

    def say(message: str, fraction: float) -> None:
        print(json.dumps({"message": message, "fraction": round(fraction, 3)}), flush=True)

    registry = Registry()
    models = [m for m in registry.manifests.values() if m.bundled_default]
    steps = len(models) + 1
    for i, m in enumerate(models):
        say(f"Downloading the {m.name} model ({i + 1} of {len(models)})…", i / steps)
        try:
            prefetch(m, lambda msg, i=i: say(msg, i / steps))
        except Exception as e:  # not fatal: it's downloaded when first needed
            say(f"Couldn't download {m.name} now ({e}); it will be when first needed", (i + 1) / steps)
    say("Getting the AI engine ready…", len(models) / steps)
    for name in ("numpy", "cv2", "torch", "torchvision", "mediapipe", "transformers", "timm", "kornia", "rawpy"):
        importlib.import_module(name)
    import rapidretouch_engine.server  # noqa: F401
    say("Ready", 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(prog="rapidretouch-engine")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="speak the JSON-lines protocol on stdin/stdout")
    sub.add_parser(
        "prepare",
        help="first-launch setup: download the default AI models and load every library once, "
        "reporting progress as JSON lines",
    )

    bd = sub.add_parser("backdrop", help="smooth the backdrop of one image")
    bd.add_argument("input")
    bd.add_argument("output")
    bd.add_argument("--mask-out", help="also save the subject mask as a PNG")
    for f in fields(backdrop_smooth.Params):
        bd.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)

    args = parser.parse_args()
    if args.command == "serve":
        from .server import serve

        serve()
        return
    if args.command == "prepare":
        prepare()
        return

    from .server import Engine

    engine = Engine(lambda msg: print(msg.get("message", msg)))
    engine.open(args.input)
    params = {f.name: getattr(args, f.name) for f in fields(backdrop_smooth.Params)}
    result = engine.export(args.output, backdrop=params)
    if args.mask_out:
        import numpy as np
        from PIL import Image

        Image.fromarray(np.round(engine.alpha * 255).astype(np.uint8)).save(args.mask_out)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

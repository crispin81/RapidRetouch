"""Command line: `retouch-engine serve` for the app, or run a tool on one file."""

from __future__ import annotations

import os

# Must be set before torch loads. Lets PyTorch grow and shrink GPU memory in
# place, so memory freed after a run can actually be reused, and returned to
# other apps sharing the card, instead of stranding in fragments.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
from dataclasses import fields

from .tools import backdrop_smooth


def main() -> None:
    parser = argparse.ArgumentParser(prog="retouch-engine")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="speak the JSON-lines protocol on stdin/stdout")

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

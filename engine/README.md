# rapidretouch-engine

The image-processing and model-inference engine of RapidRetouch. The desktop app starts
it and sends JSON-lines requests on stdin (`{"id", "method", "params"}`); it answers on
stdout and also emits `status` and `progress` events. Methods are listed in
`server.METHODS`.

```sh
uv sync
uv run pytest -q
```

- `server.py`: the engine (open, render, render_region, export, brushes, presets…)
  and its caches.
- `tools/`: one module per tool (backdrop_smooth, skin, eyes, mouth, dodge_burn,
  fabric, tone, crop, inpaint, patch, reflection, region_edit, mask_edit, scene…).
- `registry.py` + `manifests/` + `adapters/`: AI models, described by manifests
  (licence, source, checksum) and loaded through thin adapters.
- `imageio.py`: RAW (rawpy/LibRaw), TIFF and JPEG in; 16-bit TIFF and JPEG out, with
  colour profiles.
- `parallel.py`: per-pixel work split across CPU cores.

See `../docs/DEVELOPMENT_NOTES.md` for the design decisions behind the tools.

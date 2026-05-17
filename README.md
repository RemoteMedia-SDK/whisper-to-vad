# whisper-to-vad — Whisper embedding → V/A/D regressor

Standalone Path 3 Rust cdylib that registers `WhisperToVadNode` into the
[RemoteMedia SDK](https://github.com/RemoteMedia-SDK/remotemedia-sdk)
streaming pipeline registry.

This plugin runs the trained ONNX regressor from `whisper_to_vad_{ridge,mlp}.onnx`
against Whisper encoder hidden states and emits a per-input
`{valence, arousal, dominance, intensity}` JSON envelope — the listener-mode
companion to the in-tree `VadToFaceNode`.

## Use from a manifest

```json
{
  "version": "v1",
  "plugins": ["whisper-to-vad@v0.1.0"],
  "nodes": [
    {
      "id": "affect",
      "node_type": "WhisperToVadNode",
      "params": {
        "model_path": "tools/affect_calibration/artifacts/whisper_to_vad_ridge.onnx",
        "embed_dim": 1280
      }
    }
  ]
}
```

The SDK resolver expands `whisper-to-vad@v0.1.0` to
`github.com/RemoteMedia-SDK/whisper-to-vad`, fetches `plugin.toml`, then
falls through to `release-manifest.json` for the platform-specific
prebuilt `.so` / `.dylib` / `.dll` asset.

## Build the cdylib locally

```bash
git clone https://github.com/RemoteMedia-SDK/whisper-to-vad
cd whisper-to-vad
cargo build --release
# → target/release/libwhisper_to_vad_plugin.so
```

## What it exports

| Node type          | Input                                | Output                                                                 |
|--------------------|--------------------------------------|------------------------------------------------------------------------|
| `WhisperToVadNode` | `Tensor` f32 `[d]`/`[T,d]`/`[1,T,d]` | `Json{valence,arousal,dominance,intensity,source,n_frames}` |

`d` is fixed by the ONNX artifact (default 1280 for `whisper-large-v3-turbo`).
Multi-frame inputs are mean-pooled over `T` before regression (matches the
training-time pooling). Pooling mode is configurable via `params.pool` =
`"mean"` (default) or `"last"`.

## License

See `LICENSE.md`. Governed by the RemoteMedia SDK Community License 1.0.

# Changelog

## Unreleased
- feat(core): add `DetectionModel.perform_inference_batch(images, **kwargs)` with
  sequential fallback for backends.
- feat(ultralytics): real GPU-batched forward with one model call per slicing pass.
- perf(predict): single batched call per pass; converts per-image predictions and merges.
- test: parity test ensuring batched == sequential outputs on toy images.
- docs: Ultralytics integration notes; how batching works under the hood.
- chore: ruff/lint fixes and RT-DETR class cleanup.

## x.y.z
- ...
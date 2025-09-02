# UltralyticsModel

::: sahi.models.ultralytics

## Batched inference with SAHI

SAHI now calls a single `perform_inference_batch(images, **kwargs)` per slicing pass
for backends that implement it (Ultralytics does). This reduces Python overhead and
improves GPU utilization. Backends that do not implement it transparently fall back
to sequential single-image calls.

**Notes**
- You can still use all SAHI slicing/merging features.
- Output is identical to the sequential path (see parity tests).
- Tune your batch size by simply passing a list of slices; SAHI handles that internally.

No user-facing API changes are required.

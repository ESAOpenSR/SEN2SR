## What changed?

Describe the problem and the resulting behavior. Include a before/after example
for bug fixes.

## Related issue

Link the issue, if any.

## Validation

List the commands you ran and their results. For model or inference changes,
include input/output shapes, scale, dtype, and whether you tested on CPU or GPU.

```text
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 python -m pytest -q
```

## Checklist

- [ ] Targets `dev` for development, or `main` for a tested `dev` release.
- [ ] Includes regression tests for behavior changes, or explains why they do not apply.
- [ ] Updates documentation for public behavior changes, or explains why they do not apply.


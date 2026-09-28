# Supplemental MLOps artifacts — local recovery

These files come from a **separate local rerun on September 28, 2026**, using Python 3.11.15 and PyCaret 3.3.2. All 136 code cells completed without uncaught errors. They are not exports from the original hosted Colab run: that VM expired before its temporary files were recovered. The six primary notebooks remain the hosted-run evidence.

## Retained files

- `overspend_pipeline.pkl` and `overspend_api.pkl`: recovered fitted pipelines, with disabled disk cache.
- [drift_report.html](drift_report.html) and [train_test_drift.html](train_test_drift.html): computed Evidently exports.
- [plots/](plots/): classification, regression, and MLOps PNG galleries from this local run.
- [local-execution-report.json](local-execution-report.json): local-run counts, timing, and notebook hash.
- [artifact-manifest.json](artifact-manifest.json): file hashes, recovery provenance, and validation results.

Model binaries are Python pickle artifacts. Load only these trusted generated files in the matching environment; they are not portable across arbitrary library versions.

## Minimal prediction scaffold

`generated-original/` preserves PyCaret's generated API, requirements, and Dockerfile unchanged. Its requirements include the invalid constraint `pydantic<2.0.0.` and its Dockerfile targets Python 3.8. Use the corrected files in this directory instead.

The corrected API finds its model relative to the script, uses pinned dependencies and Python 3.11, requires all 18 fitted predictors, and rejects extra fields and invalid numeric inputs. **`overspent` is a current-month predictor; `overspent_next` is the future target and is rejected.** No target is supplied at prediction time.

From this directory, in a fresh Python 3.11 environment:

```sh
python -m pip install -r requirements.txt
python check_api.py
uvicorn overspend_api:app --host 127.0.0.1 --port 8000
```

`POST /predict` accepts the example household in `check_api.py` and returns a binary `prediction`. The example fields are synthetic. The endpoint is an educational local scaffold, not a hosted service.

The local HTTP check passed: a valid request produced a binary prediction; missing inputs, a future-target field, invalid household size, and nonfinite income were rejected. The Dockerfile uses Python 3.11 and copies only the API and its model; its image was **not built or deployed**. Dependencies are pinned to the tested local environment, but a clean Linux/container installation was not tested. The broader recovery environment has an unrelated `mlflow-tracing`/Pydantic conflict; that optional tracking package is not in these API requirements.

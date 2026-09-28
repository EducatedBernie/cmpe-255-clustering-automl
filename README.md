# CMPE 255 — Clustering and AutoML

Bernie Miao · Fall 2026

**Assignment package, September 28, 2026.** All six full runs finished on hosted Google Colab: **615 code cells executed, zero uncaught errors**. The notebooks preserve the actual outputs, plots, and execution counts. Video publication, remaining live execution clips, and submission verification are still pending.

## Six parts

| Part | Executed notebook | Colab runtime | Code cells | Drive copy |
|---|---|---|---:|---|
| 1 | [K-means variations](notebooks/01-kmeans-variations.ipynb) | CPU | 78 | [Open Colab](https://colab.research.google.com/drive/1mlXccGdzstrfbzD5J9NkWm_Ti86hJ-de) |
| 2 | [AutoGluon capabilities](notebooks/02-autogluon-capabilities.ipynb) | CPU | 84 | [Open Colab](https://colab.research.google.com/drive/13sugjsxIdoIncE7nM-IIFFgGH-64r-E4) |
| 3 | [AutoGluon end-to-end](notebooks/03-autogluon-end-to-end.ipynb) | CPU | 105 | [Open Colab](https://colab.research.google.com/drive/1BKO8BVRLEacSZVholMpQflodWgb82-u9) |
| 4 | [RAPIDS versus CPU](notebooks/04-rapids-vs-cpu.ipynb) | Tesla T4 | 118 | [Open Colab](https://colab.research.google.com/drive/1lBMnXMjlzkcUx2sbRPcQ5bU-sbfxGIag) |
| 5 | [PyCaret capabilities](notebooks/05-pycaret-capabilities.ipynb) | CPU | 94 | [Open Colab](https://colab.research.google.com/drive/1uCm73GPFZ0D5XlmldpEeV1Tmj5S3umqq) |
| 6 | [PyCaret MLOps](notebooks/06-pycaret-mlops.ipynb) | CPU | 136 | [Open Colab](https://colab.research.google.com/drive/13sI9xz9yIenx0nwW3YSm6Y0wDWWoJiHy) |

[Drive folder](https://drive.google.com/drive/folders/1pkQugYtane0f00zgreiaFhm1uJ85U-YZ) · [Execution reports](execution-reports/) · [Notebook hashes and counts](notebook-manifest.json)

The links come from saved sharing records. The latest saved anonymous-access checks report all six copies accessible with complete execution counts: **615/615 cells**, including the refreshed K-means copy at 78/78. These records are included in the notebook manifest; packaging itself did not perform an additional remote-access test.

## What actually ran

- **K-means:** scratch and library variants, metrics, embeddings, DEC, and compression ran. Sentence embeddings and pretrained ResNet ran; cuML and API cluster naming did not. ResNet and DEC did not beat raw pixels. A printed bad-start exercise caption contradicts its equal fitted results; the walkthrough explains this limitation.
- **AutoGluon capabilities:** tabular tasks, Chronos-Bolt, BERT fusion, MobileNetV3, MiniLM, MITRA, and scenario forecasting ran on CPU. SHAP was unavailable; permutation importance is the explanation. Quantile coverage was 75.3% for a nominal 80% interval.
- **AutoGluon end-to-end:** the main tabular lifecycle and Chronos ran. SHAP used a perturbation fallback, multimodal training used TF-IDF/logistic regression, and TabICL was unavailable. The perturbation explanation does not guarantee SHAP's additive identity.
- **RAPIDS:** cuDF, cuML, CuPy, and GPU XGBoost ran on a Tesla T4. Some operations intentionally fall back to CPU; the isolation-forest helper uses sklearn CPU code. cuGraph/PageRank was skipped because cuGraph was unavailable. Timings depend on warm-up and transfer boundaries.
- **PyCaret capabilities:** the full hosted notebook completed, including its pinned interpretation dependencies and pipeline/API-file demonstrations. Generating API files is not a deployed service.
- **PyCaret MLOps:** the full hosted run completed, including explicit Evidently drift computation and export. Optional search libraries were absent; seasonal-naive forecasting was skipped; cuML was unavailable. Missing TabPFN and TabICL used the explicitly labeled stand-in, not a foundation model. GPU-request flags do not establish GPU acceleration in this CPU run.

Zero uncaught errors means the cells completed; guarded skips and fallbacks remain part of the results.

## Rerunning and exported artifacts

K-means uses a CPU runtime. RAPIDS needs the demonstrated T4 environment. AutoGluon and PyCaret used dedicated Python 3.11 environments; PyCaret 3.3.2 uses the pinned setup cell and SHAP 0.44.1 compatibility fixes. Run setup and subsequent cells in order. The older Colab runtime selector `2025.07` was observed on September 27; menus can change, so saved Python versions and setup pins are the reproducibility reference.

The original MLOps hosted VM expired before its temporary model and HTML exports were recovered. **A separate local rerun on September 28 recovered the artifacts**, with all 136 cells completing. [Supplemental local artifacts](artifacts/mlops-local-recovery/README.md) include fitted pipelines, computed Evidently HTML reports, and plot galleries. Their provenance is local, not the same hosted run; primary notebook outputs remain unchanged.

The artifact directory preserves generated scaffold originals and supplies corrected Python 3.11 API/Docker scaffolding. Local HTTP prediction and input-validation checks passed. The future target `overspent_next` is excluded; current-month `overspent` remains a legitimate predictor. No container was built and no endpoint deployed. Printed temporary paths in notebook outputs are historical, not working links.

[checks/pycaret_shap.py](checks/pycaret_shap.py) exercises summary and correlation plots. [checks/pycaret_drift.py](checks/pycaret_drift.py) exercises Evidently's explicit computation/export API, replacing PyCaret 3.3.2's incomplete wrapper. They require the compatible notebook environment. Packaging checks their syntax; the supplemental local recovery is documented separately.

## Walkthrough video

[All six notebook walkthroughs (2:25:46)](https://youtu.be/F-qA48twpWA) — uploaded; YouTube processing and playback verification are pending.

| Part | Start |
|---|---|
| K-means | [0:00](https://youtu.be/F-qA48twpWA?t=0) |
| AutoGluon capabilities | [31:45](https://youtu.be/F-qA48twpWA?t=1905) |
| AutoGluon end-to-end | [55:04](https://youtu.be/F-qA48twpWA?t=3304) |
| RAPIDS | [1:16:49](https://youtu.be/F-qA48twpWA?t=4609) |
| PyCaret capabilities | [1:42:40](https://youtu.be/F-qA48twpWA?t=6160) |
| PyCaret MLOps | [2:00:21](https://youtu.be/F-qA48twpWA?t=7221) |

## Publication still pending

- Publish narrated walkthroughs and add verified video links. Local rendered videos are separate from this repository package.
- Finish and label the remaining live Colab clips. Code/output stills are not live execution footage.
- Verify final GitHub/Drive/video links and the course submission receipt.

Video upload and course submission remain pending.

## Sources and checks

[sources.json](sources.json) lists the six instructor links and reference SHA-256 hashes. Each hash matches its preserved download. Notebook metadata records compatibility changes; historical preparation fields may still describe the pre-execution stage. Current reports and retained outputs establish the completed hosted runs.

Packaged notebooks and reports are byte-for-byte canonical copies. [SHA256SUMS](SHA256SUMS) covers package files; [package-review.json](package-review.json) records the checks. Only notebooks, reports, source links, checks, documentation, and explicitly listed supplemental MLOps artifacts are included. Credentials, account records, raw takes, videos, environment directories, and unrelated trained models are excluded. Textual notebook content and metadata were scanned for common credential patterns; no candidates were found.

AI assistance was used for compatibility fixes, checks, and narration drafting. The approved narration voice is ElevenLabs Midhun. Instructor content is attributed above; no license or redistribution permission is inferred from its availability.

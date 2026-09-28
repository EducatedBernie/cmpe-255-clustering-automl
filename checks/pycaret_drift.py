"""Verify the explicit Evidently API used instead of PyCaret's broken wrapper."""
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report

rng = np.random.default_rng(42)
reference = pd.DataFrame({'income': rng.normal(size=100), 'target': [0, 1] * 50})
current = reference.copy()
current['income'] += 2
report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
report.run(reference_data=reference, current_data=current,
           column_mapping=ColumnMapping(target='target', prediction=None, datetime=None))
with tempfile.TemporaryDirectory(prefix='pycaret-drift-check-') as directory:
    path = Path(directory) / 'drift.html'
    report.save_html(str(path))
    assert path.stat().st_size > 1000
    assert report.as_dict()['metrics']
print('PASS: computed drift metrics and retained HTML export API')

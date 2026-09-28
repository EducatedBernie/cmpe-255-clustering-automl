"""Check the two PyCaret interpretation plots before rerunning full notebooks."""
from pathlib import Path
import os
import tempfile

import pandas as pd
import shap
from sklearn.datasets import make_classification
from pycaret.classification import ClassificationExperiment

X, y = make_classification(n_samples=80, n_features=4, n_informative=3,
                           n_redundant=0, random_state=42)
data = pd.DataFrame(X, columns=['a', 'b', 'c', 'd'])
data['target'] = y
experiment = ClassificationExperiment()
experiment.setup(data, target='target', fold=2, session_id=42,
                 n_jobs=1, html=False, verbose=False)
model = experiment.create_model('lightgbm', verbose=False)
original = Path.cwd()
with tempfile.TemporaryDirectory(prefix='pycaret-shap-check-') as directory:
    try:
        os.chdir(directory)
        experiment.interpret_model(model, plot='summary', save=True)
        experiment.interpret_model(model, plot='correlation', feature='a', save=True)
        assert len(list(Path(directory).glob('*.png'))) == 2
    finally:
        os.chdir(original)
print('PASS: summary and correlation plots; SHAP', shap.__version__)

# Legacy Entry Points

Root-level files are kept as compatibility launchers:

- `ml_prepare_features.py`
- `ml_train_meta_model.py`
- `ml_cv_validate.py`
- `ml_predict_meta_model.py`
- `ml_split_utils.py`
- `benchmark_unseen_gate.py`

Use the canonical package entry points instead:

- `python -m meta_model.prepare_features`
- `python -m meta_model.train`
- `python -m meta_model.validate`
- `python -m meta_model.predict`
- `python -m meta_model.benchmark_unseen_gate`

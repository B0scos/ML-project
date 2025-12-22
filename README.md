# End-to-End ML Project: Student Exam Score Prediction

## Overview

This repository contains an end-to-end machine learning pipeline that trains models to predict students' math scores from demographic and exam-preparation data. The project includes components for data ingestion, preprocessing, model training, evaluation, artifact management, and a minimal Flask-based prediction app skeleton.

The included dataset (`data.csv`) is a tabular dataset with features such as `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`, and exam scores (`math_score`, `reading_score`, `writing_score`). The target used in this pipeline is `math_score`.

## Features

- Data ingestion and train/test split
- Preprocessing pipeline (imputation, scaling, one-hot encoding)
- Model comparison across multiple regressors (Random Forest, Decision Tree, Gradient Boosting, Linear Regression, XGBoost, CatBoost, AdaBoost)
- Model selection by R2 score and artifact persistence (`preprocessor.pkl`, `model.pkl`) using `dill`
- Basic Flask app skeleton for serving predictions (`src/pipeline/predict_pipeline.py`)
- Structured logging and custom exception handling

## Project structure

Key files and directories:

- `data.csv` — source dataset used by the pipeline
- `artifacts/` — generated artifacts (train/test CSVs, `preprocessor.pkl`, `model.pkl`)
- `logs/` — logs created during runs
- `src/` — source package
  - `components/`
    - `data_ingestion.py` — reads `data.csv`, saves raw, splits to train/test
    - `data_transformation.py` — builds preprocessing `ColumnTransformer` and saves `preprocessor.pkl`
    - `model_trainer.py` — trains several regressors, evaluates them, and saves the best model as `model.pkl`
  - `pipeline/`
    - `train_pipeline.py` — (placeholder)
    - `predict_pipeline.py` — minimal Flask app (index route only)
  - `utils.py` — helpers: `save_object`, `evaluate_models`
  - `exception.py` — custom exception wrapper
  - `logger.py` — logging setup
- `requirements.txt` — Python dependencies
- `setup.py` — package install script

## Requirements

- Python 3.8+
- See `requirements.txt` for packages. Notable dependencies include:
  - pandas, numpy, scikit-learn
  - catboost, xgboost
  - Flask, dill

## Installation

1. Create and activate a virtual environment (Windows PowerShell example):

```powershell
python -m venv venv
& .\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
# or install package in editable mode
pip install -e .
```

## How to run

### Full ingestion → transformation → training flow

The `data_ingestion` module's `__main__` block runs the basic flow: it reads `data.csv`, saves the raw file to `artifacts/data.csv`, splits into train/test, runs data transformation, trains models, and saves the best model.

From the project root run:

```powershell
python -m src.components.data_ingestion
```

After the script finishes you should see:

- `artifacts/train.csv`, `artifacts/test.csv`
- `artifacts/preprocessor.pkl` (preprocessing pipeline)
- `artifacts/model.pkl` (saved best model)
- New logs in the `logs/` directory

### Running the Flask app (development)

The Flask app is defined in `src/pipeline/predict_pipeline.py` as `app`. It currently exposes only a root index route. To run the app using Flask CLI:

Windows PowerShell example:

```powershell
$env:FLASK_APP = 'src.pipeline.predict_pipeline'
flask run
```

Visit http://127.0.0.1:5000/ to see the basic index response. The `/predict` endpoint is not implemented yet — see the extension suggestions below if you want to add it.

## How it works (component details)

- Data Ingestion (`src/components/data_ingestion.py`): reads `data.csv`, writes raw and splits using `train_test_split(test_size=0.2, random_state=42)`.
- Data Transformation (`src/components/data_transformation.py`):
  - Numerical pipeline: median imputation + `StandardScaler` for `writing_score` and `reading_score`.
  - Categorical pipeline: most-frequent imputation + `OneHotEncoder` for categorical columns.
  - The preprocessor is saved to `artifacts/preprocessor.pkl`.
- Model Trainer (`src/components/model_trainer.py`):
  - Trains multiple regressors and evaluates their R2 score using `evaluate_models`.
  - The best model is saved at `artifacts/model.pkl`.

## Extending the API with a `/predict` endpoint (example)

To add a prediction endpoint, you can implement a route that:

1. Loads the saved `preprocessor.pkl` and `model.pkl` using `dill`.
2. Accepts JSON input with the same fields used for training (all features except `math_score`).
3. Applies preprocessing and returns the predicted `math_score`.

Minimal example (to be added to `src/pipeline/predict_pipeline.py`):

```python
# Example: add this to `predict_pipeline.py`
from flask import request
import dill

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json()
    # payload should be a dict containing the feature columns
    df = pd.DataFrame([payload])
    preprocessor = dill.load(open('artifacts/preprocessor.pkl', 'rb'))
    model = dill.load(open('artifacts/model.pkl', 'rb'))
    X = preprocessor.transform(df)
    y_pred = model.predict(X)
    return jsonify({'math_score_pred': float(y_pred[0])})
```

Be sure to validate inputs and handle missing or malformed requests in production code.

## Evaluation and metrics

- Models are compared using R2 score (`sklearn.metrics.r2_score`) in `src/utils.py`.
- The project prints and logs the best model's name and score when training finishes.

## Issues, improvements and next steps

- Add a complete `train_pipeline.py` orchestrator and CLI entry points.
- Implement and test the `/predict` endpoint in the Flask app.
- Add unit and integration tests (pytest) for components and the API.
- Add CI (GitHub Actions) to run tests and checks on PRs.
- Containerize the app with Docker for consistent deployment.

## Contributing

If you plan to contribute, please open an issue describing the change or improvement, then submit a pull request with tests and documentation updates.


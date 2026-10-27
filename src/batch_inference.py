from pathlib import Path

import mlflow
import mlflow.pyfunc
from sklearn.metrics import roc_auc_score

from common import MODEL_NAME, connect
from data import load_wine_data, split

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def main():
    connect()
    X_train, X_val, X_test, y_train, y_val, y_test = split(load_wine_data())

    model_uri = f"models:/{MODEL_NAME}/production"
    model = mlflow.pyfunc.load_model(model_uri)
    scores = model.predict(X_test)
    print(f"Loaded {model_uri}: test AUC = {roc_auc_score(y_test, scores):.4f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    out = X_test.assign(prediction=scores, actual=y_test.values)
    out.to_csv(OUTPUT_DIR / "batch_predictions.csv", index=False)
    print(f"Wrote {len(out)} predictions to {OUTPUT_DIR / 'batch_predictions.csv'}")


if __name__ == "__main__":
    main()


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

app = FastAPI()

# ---------------------------------
# Load existing model and artifacts
# ---------------------------------
model = tf.keras.models.load_model("cnn_multiK_model_best.h5")

scaler = joblib.load("scaler.pkl")
feature_list = joblib.load("feature_list.pkl")

proto_map = joblib.load("proto_smooth_map.pkl")
state_map = joblib.load("state_smooth_map.pkl")

label_classes = joblib.load("label_encoder_classes.pkl")
global_mean = joblib.load("global_mean.pkl")


# ---------------------------------
# Input schemas
# ---------------------------------
class InputData(BaseModel):
    features: dict


class BatchInput(BaseModel):
    records: list[dict]


# ---------------------------------
# Shared preprocessing function
# ---------------------------------
def preprocess_records(records):

    if not records:
        raise ValueError("No feature records received.")

    processed_rows = []

    for record in records:

        row = record.copy()

        # Encode categorical features
        row["proto"] = proto_map.get(
            row.get("proto"), global_mean
        )

        row["state"] = state_map.get(
            row.get("state"), global_mean
        )

        # Preserve existing missing-feature handling
        for feature in feature_list:
            if feature not in row:
                row[feature] = 0.0

        processed_rows.append(
            [row[feature] for feature in feature_list]
        )

    X_df = pd.DataFrame(
        processed_rows,
        columns=feature_list
    )

    # Scale all records together
    X_scaled = scaler.transform(X_df)

    # CNN input: (N, 1, 34, 1)
    X_scaled = X_scaled.reshape(
        len(records), 1, 34, 1
    )

    return X_scaled


# ---------------------------------
# Existing single-flow endpoint
# ---------------------------------
@app.post("/predict")
def predict(data: InputData):

    try:
        X = preprocess_records([data.features])

        pred = model.predict(X, verbose=0)

        class_id = int(np.argmax(pred[0]))
        label = str(label_classes[class_id])
        confidence = float(np.max(pred[0]))

        return {
            "class_id": class_id,
            "label": label,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ---------------------------------
# New batch prediction endpoint
# ---------------------------------
@app.post("/predict_batch")
def predict_batch(data: BatchInput):

    if not data.records:
        raise HTTPException(
            status_code=400,
            detail="Batch cannot be empty."
        )

    try:
        X = preprocess_records(data.records)

        # One CNN inference call for the entire batch
        predictions = model.predict(X, verbose=0)

        results = []

        for prediction in predictions:

            class_id = int(np.argmax(prediction))
            label = str(label_classes[class_id])
            confidence = float(np.max(prediction))

            results.append({
                "class_id": class_id,
                "label": label,
                "confidence": confidence
            })

        if len(results) != len(data.records):
            raise RuntimeError(
                "Prediction count does not match input count."
            )

        print(
            f"[BATCH PREDICTION] "
            f"Received: {len(data.records)}, "
            f"Predicted: {len(results)}"
        )

        return {
            "received": len(data.records),
            "predicted": len(results),
            "predictions": results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

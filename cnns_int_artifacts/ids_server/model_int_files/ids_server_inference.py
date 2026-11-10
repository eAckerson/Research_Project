from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import threading

app = FastAPI(title="AnDO CNN IDS Output Server")

# ---------------------------------
# Output configuration
# ---------------------------------

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_CSV = OUTPUT_DIR / "CNN_Predictions.csv"
FEATURES_CSV = OUTPUT_DIR / "AnDO_ONOS_Features.csv"

# Protect CSV files against concurrent writes.
csv_lock = threading.Lock()


# ---------------------------------
# Input schemas
# ---------------------------------

class BatchInput(BaseModel):
    records: list[dict]


# ---------------------------------
# CSV append function
# ---------------------------------

def append_records_to_csv(records, csv_path):

    if not records:
        raise ValueError("No records supplied.")

    new_df = pd.DataFrame(records)

    with csv_lock:

        if csv_path.exists() and csv_path.stat().st_size > 0:

            # Preserve the existing CSV column structure.
            existing_columns = pd.read_csv(
                csv_path,
                nrows=0
            ).columns.tolist()

            # Add any newly encountered columns.
            new_columns = [
                col for col in new_df.columns
                if col not in existing_columns
            ]

            if new_columns:
                existing_df = pd.read_csv(csv_path)
                existing_df = existing_df.reindex(
                    columns=existing_columns + new_columns
                )

                new_df = new_df.reindex(
                    columns=existing_columns + new_columns
                )

                existing_df.to_csv(
                    csv_path,
                    index=False
                )

            else:
                new_df = new_df.reindex(
                    columns=existing_columns
                )

            new_df.to_csv(
                csv_path,
                mode="a",
                header=False,
                index=False
            )

        else:
            new_df.to_csv(
                csv_path,
                index=False
            )

    return len(new_df)


# ---------------------------------
# Health/status endpoint
# ---------------------------------

@app.get("/")
def home():

    return {
        "status": "running",
        "service": "AnDO CNN IDS Output Server",
        "prediction_csv": str(PREDICTIONS_CSV),
        "feature_csv": str(FEATURES_CSV)
    }


# ---------------------------------
# Receive CNN predictions
# ---------------------------------

@app.post("/predictions/batch")
def receive_predictions(data: BatchInput):

    if not data.records:
        raise HTTPException(
            status_code=400,
            detail="Prediction batch cannot be empty."
        )

    try:
        records = []

        for prediction in data.records:

            record = prediction.copy()

            # Add a server receipt timestamp if absent.
            record.setdefault(
                "server_received_at",
                datetime.now(timezone.utc).isoformat()
            )

            records.append(record)

        count = append_records_to_csv(
            records,
            PREDICTIONS_CSV
        )

        print(
            f"[CNN OUTPUT] Received and stored "
            f"{count} predictions."
        )

        for record in records:
            print(
                f"Prediction: "
                f"class_id={record.get('class_id')}, "
                f"label={record.get('label')}, "
                f"confidence={record.get('confidence')}"
            )

        return {
            "status": "success",
            "received": count,
            "stored": count,
            "file": str(PREDICTIONS_CSV)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ---------------------------------
# Receive AnDO/ONOS features
# ---------------------------------

@app.post("/features/batch")
def receive_features(data: BatchInput):

    if not data.records:
        raise HTTPException(
            status_code=400,
            detail="Feature batch cannot be empty."
        )

    try:
        count = append_records_to_csv(
            data.records,
            FEATURES_CSV
        )

        print(
            f"[AnDO/ONOS OUTPUT] "
            f"Received and stored {count} feature records."
        )

        return {
            "status": "success",
            "received": count,
            "stored": count,
            "file": str(FEATURES_CSV)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


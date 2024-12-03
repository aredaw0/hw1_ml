from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("models/model.pkl")

class Item(BaseModel):
    year: int
    km_driven: int
    mileage: float
    engine: float
    max_power: float
    seats: float


@app.post("/predict_item")
def predict_item(item: Item):
    data = pd.DataFrame([item.dict()])
    prediction = model.predict(data)
    return {"predicted_price": prediction[0]}


@app.post("/predict_items_csv")
async def predict_items_csv(file: UploadFile):
    df = pd.read_csv(file.file)
    required_columns = ["year", "km_driven", "mileage", "engine", "max_power", "seats"]

    if not all(col in df.columns for col in required_columns):
        raise HTTPException(status_code=400, detail='///')

    predictions = model.predict(df[required_columns])
    df["predicted_price"] = predictions

    output_file = "predicted_results.csv"
    df.to_csv(output_file, index=False)

    return FileResponse(output_file, media_type="text/csv", filename="predicted_results.csv")

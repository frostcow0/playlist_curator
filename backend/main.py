"""Author: Jon Martin

Utilizes FastAPI to serve user's songs and recommendations"""

import uvicorn
import pandas as pd
from fastapi import FastAPI
from recommendation_engine import get_user_data, get_recommendations
from pydantic import BaseModel
import logging


class SongData(BaseModel):
    feature_saved: str
    feature_50: str

class Recommendations(BaseModel):
    formatted: dict
    raw: dict

app = FastAPI()

logging.basicConfig(level=logging.INFO)

@app.get("/user_data")
def user_data() -> dict:
    data = get_user_data()
    return data

@app.post("/minkowski_recommend")
def mink_recommend(song_dict:SongData) -> dict:
    frames = {
        "feature_saved": pd.read_json(song_dict.feature_saved),
        "feature_50": pd.read_json(song_dict.feature_50),
    }
    formatted, raw = get_recommendations(song_data=frames, method="minkowski")
    return Recommendations(formatted=formatted.to_dict(), raw=raw.to_dict())

@app.post("/test_post/{text}")
def test_post(text:str):
    print(text)
    return text


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)

"""Load the saved native model together with its training-fitted encoder."""
import json
from pathlib import Path
import xgboost as xgb
from .data import Encoder


def predict(frame, run_directory, nthread=4):
    root = Path(run_directory)
    encoder = Encoder(**json.loads((root/"encoder.json").read_text()))
    model = xgb.Booster(params={"nthread": nthread})
    model.load_model(root/"model.ubj")
    return model.predict(encoder.matrix(frame, nthread=nthread))

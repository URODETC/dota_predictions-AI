import xgboost as xgb
import torch
import torch.nn as nn
import joblib
from gbnet.xgbmodule import XGBModule
import numpy as np
from itertools import *


def time_strenght(team, duration, duration_ind=0):
    if duration_ind == 0:
        if duration < 25:
            values = np.array(
                [hero_stats_time.get(a, {}).get(1, 0.0) for a in team], dtype=np.float32
            )
        elif duration >= 25 and duration < 30:
            values = np.array(
                [hero_stats_time.get(a, {}).get(2, 0.0) for a in team], dtype=np.float32
            )
        elif duration >= 30 and duration < 32.5:
            values = np.array(
                [hero_stats_time.get(a, {}).get(3, 0.0) for a in team], dtype=np.float32
            )
        elif duration >= 32.5 and duration < 35:
            values = np.array(
                [hero_stats_time.get(a, {}).get(4, 0.0) for a in team], dtype=np.float32
            )
        elif duration >= 35 and duration < 37.5:
            values = np.array(
                [hero_stats_time.get(a, {}).get(5, 0.0) for a in team], dtype=np.float32
            )
        elif duration >= 37.5 and duration < 40:
            values = np.array(
                [hero_stats_time.get(a, {}).get(6, 0.0) for a in team], dtype=np.float32
            )
        elif duration >= 40 and duration < 50:
            values = np.array(
                [hero_stats_time.get(a, {}).get(7, 0.0) for a in team], dtype=np.float32
            )
        elif duration >= 50:
            values = np.array(
                [hero_stats_time.get(a, {}).get(8, 0.0) for a in team], dtype=np.float32
            )
    else:
        values = np.array(
            [hero_stats_time.get(a, {}).get(duration_ind, 0.0) for a in team],
            dtype=np.float32,
        )
    return values.mean()


def counter_synergy(radiant, dire):
    values = np.fromiter(
        (matchup_synergy.get((a, b), 0.0) for a, b in product(radiant, dire)),
        dtype=np.float32,
    )
    return values.mean()


def synergy(team: list):
    team = np.sort(np.array(team, dtype=np.int16))
    pairs = np.array(list(combinations(team, 2)), dtype=np.int16)
    values = np.fromiter(
        (pair_synergy.get((a, b), 0.5) for a, b in pairs),
        dtype=np.float32,
        count=len(pairs),
    )
    return values.mean()


artifacts = joblib.load("model/artifacts.pkl")

booster = xgb.Booster()
booster.load_model("model/xgb.json")


class XGBWrapper:
    def __init__(self, booster):
        self.booster = booster

    def __call__(self, x):
        dmat = xgb.DMatrix(x)
        return self.booster.predict(dmat)


xgb_model = XGBWrapper(booster)


class ForecastModule(nn.Module):
    def __init__(self, dimensions, xgb_module):
        super().__init__()
        self.linear = nn.Linear(dimensions, 1)
        self.xgb = xgb_module

    def forward(self, t):
        linear_out = self.linear(t)

        if getattr(self.xgb, "is_trained", False):
            t_np = t.detach().cpu().numpy()
            xgb_out_np = self.xgb(t_np)
            xgb_out = torch.tensor(xgb_out_np, dtype=torch.float32, device=t.device)
        else:
            xgb_out = torch.zeros(t.size(0), 1, device=t.device)

        return linear_out + xgb_out


model = ForecastModule(10, xgb_model)
params = torch.load("model/linear.pth", map_location="cpu")
model.linear.load_state_dict(params)

carry_matchup = artifacts["carry_matchup"]
matchup_synergy = artifacts["matchup_synergy"]
mid_matchup = artifacts["mid_matchup"]
offlane_matchup = artifacts["offlane_matchup"]
pair_synergy = artifacts["pair_synergy"]
hero_stats_time = artifacts["hero_stats_time"]
sup_synergy = artifacts["sup_synergy"]


def prediction(radiant, dire, model):
    r_synergy_val = synergy(radiant)
    d_synergy_val = synergy(dire)
    csynergy_val = counter_synergy(radiant, dire)
    carry = carry_matchup[radiant[0], dire[0]]
    mid = mid_matchup[radiant[1], dire[1]]
    offlane = offlane_matchup[radiant[2], dire[2]]
    r_sup = sup_synergy[radiant[3], radiant[4]]
    d_sup = sup_synergy[dire[3], dire[4]]

    results = []

    for duration in range(1, 9):
        input_features = [
            r_synergy_val,
            d_synergy_val,
            csynergy_val,
            time_strenght(radiant, duration, duration),
            time_strenght(dire, duration, duration),
            carry,
            mid,
            offlane,
            r_sup,
            d_sup,
        ]

        X_tensor = torch.tensor([input_features], dtype=torch.float32)
        with torch.no_grad():
            pred_tensor = model(X_tensor)
            prob_radiant = torch.sigmoid(pred_tensor).item()
            prob_dire = 1 - prob_radiant

        results.append({"Radiant": prob_radiant, "Dire": prob_dire, "Time": duration})

    return results


print(prediction([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], model))

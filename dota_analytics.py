import numpy as np
from itertools import combinations, product
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

print("\nLoading data...")
players2025 = pd.read_csv("players2025.csv")
players202601 = pd.read_csv("players202601.csv")
main_metadata2025 = pd.read_csv("main_metadata2025.csv")
main_metadata202601 = pd.read_csv("main_metadata202601.csv")
heroesdf = pd.read_csv("Constants.Heroes.csv")

players = pd.concat([players2025, players202601])
matches = pd.concat([main_metadata2025, main_metadata202601])

heroes = heroesdf.melt(id_vars="id", value_vars="name")
heroes = dict(zip(heroes["id"], heroes["value"]))
players["hero_name"] = players["hero_id"].apply(lambda x: heroes.get(x))

attack_type = heroesdf.melt(id_vars="id", value_vars="attack_type")
attack_type = dict(zip(attack_type["id"], attack_type["value"]))
players["hero_attack_type"] = players["hero_id"].apply(lambda x: attack_type.get(x))
print("Data loading complete. Preparing training dataset...")


def synergy(team: list):
    team = np.sort(np.array(team, dtype=np.int16))
    pairs = np.array(list(combinations(team, 2)), dtype=np.int16)
    values = np.fromiter(
        (pair_synergy.get((a, b), 0.5) for a, b in pairs),
        dtype=np.float32,
        count=len(pairs),
    )
    return values.mean()


def counter_synergy(radiant, dire):
    values = np.fromiter(
        (matchup_synergy.get((a, b), 0.0) for a, b in product(radiant, dire)),
        dtype=np.float32,
    )
    return values.mean()


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


"""
data [r1,r2,r3,r4,r5,d1,d2,d3,d4,d5]
target win (0 - radiant 1 dire)
"""
df = players.merge(matches, left_on="match_id", right_on="match_id", how="left")
df.rename(columns={"duration_x": "duration"}, inplace=True)
df.rename(columns={"radiant_win_x": "radiant_win"}, inplace=True)
df["duration"] = df["duration"] / 60
df["isRadiant"] = np.where(df["isRadiant"] == True, "1", "0").astype(int)
df.rename(columns={"isRadiant": "teams"}, inplace=True)
df["radiant_win"] = np.where(df["radiant_win"] == True, "1", "0").astype(int)
df["ranged"] = np.where(df["hero_attack_type"] == "Ranged", "1", "0").astype(int)

df = df[
    [
        "match_id",
        "hero_id",
        "hero_name",
        "ranged",
        "player_slot",
        "teams",
        "win",
        "radiant_win",
        "duration",
    ]
]

df["id"] = df.groupby("match_id").ngroup()
df

data = df.assign(
    player_slot=df["player_slot"] + (df.groupby(["id", "player_slot"]).cumcount() + 1)
).pivot_table(index="id", columns="player_slot", values="hero_id")
data.columns = range(data.shape[1])
data.columns = ["r_1", "r_2", "r_3", "r_4", "r_5", "d_1", "d_2", "d_3", "d_4", "d_5"]
data = data.merge(
    df[["id", "duration"]].drop_duplicates(subset=["id"]), on="id", how="left"
)
target = df.groupby("id").min()[["radiant_win"]]
teams_data = []
for index, row in data.iterrows():
    teams_data.append(
        [row[["r_1", "r_2", "r_3", "r_4", "r_5"]].astype(int).tolist()]
        + [row[["d_1", "d_2", "d_3", "d_4", "d_5"]].astype(int).tolist()]
        + [target.loc[index, "radiant_win"]]
        + [row["duration"]]
    )
teams = pd.DataFrame(teams_data, columns=["radiant", "dire", "radiant_win", "duration"])

pair_stats = {}

for i, row in teams.iterrows():
    for a, b in combinations(sorted(row["radiant"]), r=2):
        if (a, b) not in pair_stats:
            pair_stats[(a, b)] = {"matches": 0, "wins": 0}
        pair_stats[(a, b)]["matches"] += 1
        if row["radiant_win"] == 1:
            pair_stats[(a, b)]["wins"] += 1

    for a, b in combinations(sorted(row["dire"]), r=2):
        if (a, b) not in pair_stats:
            pair_stats[(a, b)] = {"matches": 0, "wins": 0}
        pair_stats[(a, b)]["matches"] += 1
        if row["radiant_win"] == 0:
            pair_stats[(a, b)]["wins"] += 1

pair_synergy = {}

for pair, stats in pair_stats.items():
    if stats["matches"] >= 10:
        pair_synergy[pair] = stats["wins"] / stats["matches"]

matchup_stats = {}
hero_stats = {}
for i, row in teams.iterrows():
    radiant = row["radiant"]
    dire = row["dire"]
    radiant_win = row["radiant_win"]

    for hero in radiant:
        hero_stats.setdefault(hero, {"matches": 0, "wins": 0})
        hero_stats[hero]["matches"] += 1
        if radiant_win == 1:
            hero_stats[hero]["wins"] += 1

    for hero in dire:
        hero_stats.setdefault(hero, {"matches": 0, "wins": 0})
        hero_stats[hero]["matches"] += 1
        if radiant_win == 0:
            hero_stats[hero]["wins"] += 1

    for a, b in product(radiant, dire):
        pair = (a, b)
        if pair not in matchup_stats:
            matchup_stats[pair] = {"matches": 0, "wins": 0}
        matchup_stats[pair]["matches"] += 1
        if radiant_win == 1:
            matchup_stats[pair]["wins"] += 1

matchup_synergy = {}

for (a, b), stats in matchup_stats.items():
    w, m = stats["wins"], stats["matches"]

    if m == 0 or hero_stats[a]["matches"] == 0:
        continue

    matchup_synergy[(a, b)] = w / m - hero_stats[a]["wins"] / hero_stats[a]["matches"]

hero_stats_time = {}

for _, row in teams.iterrows():
    radiant = row["radiant"]
    dire = row["dire"]
    radiant_win = row["radiant_win"]
    game_time = row["duration"]

    time_category = None
    if game_time < 25:
        time_category = 1
    elif 25 <= game_time < 30:
        time_category = 2
    elif 30 <= game_time < 32.5:
        time_category = 3
    elif 32.5 <= game_time < 35:
        time_category = 4
    elif 35 <= game_time < 37.5:
        time_category = 5
    elif 37.5 <= game_time < 40:
        time_category = 6
    elif 40 <= game_time < 50:
        time_category = 7
    elif game_time >= 50:
        time_category = 8

    if time_category:
        for hero in radiant:
            hero_stats_time.setdefault(
                hero,
                {
                    1: {"matches": 0, "wins": 0},
                    2: {"matches": 0, "wins": 0},
                    3: {"matches": 0, "wins": 0},
                    4: {"matches": 0, "wins": 0},
                    5: {"matches": 0, "wins": 0},
                    6: {"matches": 0, "wins": 0},
                    7: {"matches": 0, "wins": 0},
                    8: {"matches": 0, "wins": 0},
                },
            )
            hero_stats_time[hero][time_category]["matches"] += 1
            if radiant_win == 1:
                hero_stats_time[hero][time_category]["wins"] += 1

        for hero in dire:
            hero_stats_time.setdefault(
                hero,
                {
                    1: {"matches": 0, "wins": 0},
                    2: {"matches": 0, "wins": 0},
                    3: {"matches": 0, "wins": 0},
                    4: {"matches": 0, "wins": 0},
                    5: {"matches": 0, "wins": 0},
                    6: {"matches": 0, "wins": 0},
                    7: {"matches": 0, "wins": 0},
                    8: {"matches": 0, "wins": 0},
                },
            )
            hero_stats_time[hero][time_category]["matches"] += 1
            if radiant_win == 0:
                hero_stats_time[hero][time_category]["wins"] += 1

for hero, times_stats in hero_stats_time.items():
    for time_cat, stats in times_stats.items():
        matches = stats["matches"]
        wins = stats["wins"]
        if matches > 0:
            overall_win_rate = (
                hero_stats[hero]["wins"] / hero_stats[hero]["matches"]
                if hero_stats[hero]["matches"] > 0
                else 0
            )
            hero_stats_time[hero][time_cat] = (wins / matches) - overall_win_rate
        else:
            hero_stats_time[hero][time_cat] = 0.0

hero_own_st = list(hero_stats_time[59].items())
for i in range(len(hero_own_st)):
    hero_own_st[i] = list(hero_own_st[i])
    game_time = hero_own_st[i][0]
    time_category = None
    if game_time == 1:
        time_category = 25
    elif game_time == 2:
        time_category = 30
    elif game_time == 3:
        time_category = 32.5
    elif game_time == 4:
        time_category = 35
    elif game_time == 5:
        time_category = 37.5
    elif game_time == 6:
        time_category = 40
    elif game_time == 7:
        time_category = 50
    elif game_time == 8:
        time_category = 60
    hero_own_st[i][0] = time_category

train = pd.DataFrame()

train["r_synergy"] = data[["r_1", "r_2", "r_3", "r_4", "r_5"]].apply(
    lambda x: synergy(x.dropna().tolist()), axis=1
)
train["d_synergy"] = data[["d_1", "d_2", "d_3", "d_4", "d_5"]].apply(
    lambda x: synergy(x.dropna().tolist()), axis=1
)

train["r_time"] = data.apply(
    lambda x: time_strenght(
        x[["r_1", "r_2", "r_3", "r_4", "r_5"]].dropna().tolist(), x["duration"]
    ),
    axis=1,
)
train["d_time"] = data.apply(
    lambda x: time_strenght(
        x[["d_1", "d_2", "d_3", "d_4", "d_5"]].dropna().tolist(), x["duration"]
    ),
    axis=1,
)

train["csynergy"] = data.apply(
    lambda row: counter_synergy(
        row[["r_1", "r_2", "r_3", "r_4", "r_5"]].dropna().tolist(),
        row[["d_1", "d_2", "d_3", "d_4", "d_5"]].dropna().tolist(),
    ),
    axis=1,
)
print("Training dataset prepared.")
print(f"Training dataset shape: {train.shape}")
y = target
X = train


def evaluate_lgbm(params):
    params["objective"] = "binary"
    params["metric"] = "binary_logloss"
    params["boosting_type"] = "gbdt"
    params["verbosity"] = -1
    params["n_jobs"] = -1
    params["device"] = "gpu"

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = (
            y.iloc[train_idx].values.ravel(),
            y.iloc[val_idx].values.ravel(),
        )

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_error",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        aucs.append(auc)

    return np.mean(aucs)


def objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
        "n_estimators": 1000,
    }

    return evaluate_lgbm(params)


print("Starting model optimization...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=500)
print("Model optimization complete.")
best_params = study.best_params.copy()
best_params.update(
    {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "n_jobs": -1,
    }
)

final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X, y)
proba = final_model.predict_proba(X)[:, 1]
pred = final_model.predict(X)
acc = accuracy_score(y, pred)
auc = roc_auc_score(y, proba)
print(f"\n\nFinal Accuracy: {acc:.4f}")
print(f"Final AUC: {auc:.4f}")
print("\n--- Best Hyperparams ---")
for k, v in best_params.items():
    print(f"{k}: {v}")

joblib.dump(final_model, "model.pkl")
joblib.dump(hero_stats_time, "hero_stats_time.pkl")
joblib.dump(matchup_synergy, "matchup_synergy.pkl")
joblib.dump(pair_synergy, "pair_synergy.pkl")

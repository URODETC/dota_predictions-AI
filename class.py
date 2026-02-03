import numpy as np
from itertools import combinations, product



class PredicitionModel:
    def __init__(self, carry, mid, offlane, synergy, matchup_synergy, time_strenght, model):
        self.carry = carry
        self.mid = mid
        self.offlane = offlane
        self.pair_synergy = synergy
        self.matchup_synergy = matchup_synergy
        self.time_strenght = time_strenght
        self.model = model

    def synergy(self, team: list):
        team = np.sort(np.array(team, dtype=np.int16))
        pairs = np.array(list(combinations(team, 2)), dtype=np.int16)
        values = np.fromiter(
            (self.pair_synergy.get((a, b), 0.5) for a, b in pairs),
            dtype=np.float32,
            count=len(pairs)
        )
        return values.mean()

    def counter_synergy(self, radiant:list, dire:list):
        values = np.fromiter(
            (self.matchup_synergy.get((a, b), 0.0) for a, b in product(radiant, dire)),
            dtype=np.float32
        )
        return values.mean()

    def time_strenght(team, duration, duration_ind=0):
        if duration_ind == 0:
            if duration < 25:
                values = np.array([hero_stats_time.get(a, {}).get(1, 0.0) for a in team], dtype=np.float32)
            elif duration >= 25 and duration < 30:
                values = np.array([hero_stats_time.get(a, {}).get(2, 0.0) for a in team], dtype=np.float32)
            elif duration >= 30 and duration < 32.5:
                values = np.array([hero_stats_time.get(a, {}).get(3, 0.0) for a in team], dtype=np.float32)
            elif duration >= 32.5 and duration < 35:
                values = np.array([hero_stats_time.get(a, {}).get(4, 0.0) for a in team], dtype=np.float32)
            elif duration >= 35 and duration < 37.5:
                values = np.array([hero_stats_time.get(a, {}).get(5, 0.0) for a in team], dtype=np.float32)
            elif duration >= 37.5 and duration < 40:
                values = np.array([hero_stats_time.get(a, {}).get(6, 0.0) for a in team], dtype=np.float32)
            elif duration >= 40 and duration < 50:
                values = np.array([hero_stats_time.get(a, {}).get(7, 0.0) for a in team], dtype=np.float32)
            elif duration >= 50:
                values = np.array([hero_stats_time.get(a, {}).get(8, 0.0) for a in team], dtype=np.float32)
        else:
            values = np.array([hero_stats_time.get(a, {}).get(duration_ind, 0.0) for a in team], dtype=np.float32)
        return values.mean()
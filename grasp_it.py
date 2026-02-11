import math
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class Rule:
    sL: set = field(default_factory=set)
    sR: set = field(default_factory=set)
    rt: str = ""
# degats de la tempete : 
# Sl = Signature(Degats)(set) = [h:Dommages ; h:destruction : ; type_semantique : A ; regle_existante : regle]
# sR = Signature (tempete)(set)
# rt : r_has_causatif
# Calcul similiratié features
def cosine_sim(s1, s2):
    if not s1 or not s2:
        return 0.0
    intersection = s1 & s2
    if not intersection:
        return 0.0
    return len(intersection) / (math.sqrt(len(s1)) * math.sqrt(len(s2)))
# Degats : 25 termes , destruction : 37 , intersection : 22 
# score 22 / (racine(25)*racine(37)
# modèle GRASPIT , classification par proches voisins
class GRASPit:
    def __init__(self, extractor, k=5):
        self.extractor = extractor
        self.k = k # à tester avec autres k 
        self.rules = []
    #stockage règles corpus
    def train(self, train_data):
        self.rules = []
        total = len(train_data)
        for i, (a, b, rt) in enumerate(train_data):
            set_a, set_b = self.extractor.extract_pair(a, b)
            #mot inconnu 
            if not set_a and not set_b:
                continue
            self.rules.append(Rule(sL=set_a, sR=set_b, rt=rt))
            if i+1 == total:
                print(f"  [{i+1}/{total}] Regles creees")

    def score_r(self, a, b):
        set_a, set_b = self.extractor.extract_pair(a, b)
        scored = []
        for rule in self.rules:
            sim_l = cosine_sim(set_a, rule.sL)
            sim_r = cosine_sim(set_b, rule.sR)
            score = (sim_l + sim_r) / 2.0
            scored.append((score, rule.rt, sim_l, sim_r))
        scored.sort(key=lambda x: -x[0])
        return scored

    #knn 
    def predict(self, a, b, top_n=1):
        scored = self.score_r(a, b)

        votes = defaultdict(float)
        for score, rt, _, _ in scored[:self.k]:
            votes[rt] += score * score

        if not votes:
            return ("unknown", 0.0) if top_n == 1 else []

        ranked = sorted(votes.items(), key=lambda x: -x[1])
        if top_n == 1:
            return ranked[0]
        return ranked[:top_n]

    #knn pour mode interactif
    def predict_2(self, a, b, top_n=3):
        scored = self.score_r(a, b)

        votes = defaultdict(lambda: {"score": 0.0, "count": 0, "best_sim_A": 0, "best_sim_B": 0})
        for score, rt, sim_l, sim_r in scored[:self.k]:
            votes[rt]["score"] += score
            votes[rt]["count"] += 1
            if score > votes[rt].get("best_score", 0):
                votes[rt]["best_sim_A"] = sim_l
                votes[rt]["best_sim_B"] = sim_r
                votes[rt]["best_score"] = score

        type_scores = {}
        for rt, info in votes.items():
            type_scores[rt] = {
                "score": info["score"],
                "sim_A": info["best_sim_A"],
                "sim_B": info["best_sim_B"],
                "votes": info["count"],
            }

        ranked = sorted(type_scores.items(), key=lambda x: -x[1]["score"])
        return ranked[:top_n]

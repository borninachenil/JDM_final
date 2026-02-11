"""
GRASP-it : Genitive Relations And Semantic Pattern Identification Tool.

Algorithme d'apprentissage de regles pour la classification des relations
semantiques dans les constructions genitives "A de B".

Deux modes :
- "fusion" : GRASP-it classique avec fusion de regles (article original)
- "knn" : approche k-NN directe sur les signatures brutes (plus performant)
"""

import math
from dataclasses import dataclass, field
from collections import defaultdict

from config import FUSION_THRESHOLD
from signature import SignatureExtractor


@dataclass
class Rule:
    """Regle de classification : paire de contraintes + type de relation."""
    sL: set = field(default_factory=set)   # Signature de A
    sR: set = field(default_factory=set)   # Signature de B
    rt: str = ""                            # Type de relation
    weight: float = 1.0                     # Fiabilite (nombre de fusions + 1)


def cosine_sim(s1, s2, idf=None):
    """Similarite cosinus entre deux ensembles, optionnellement ponderee par IDF."""
    if not s1 or not s2:
        return 0.0
    intersection = s1 & s2
    if not intersection:
        return 0.0
    if idf is None:
        return len(intersection) / (math.sqrt(len(s1)) * math.sqrt(len(s2)))
    # Cosinus pondere par IDF
    dot = sum(idf.get(f, 1.0) for f in intersection)
    norm1 = math.sqrt(sum(idf.get(f, 1.0) ** 2 for f in s1))
    norm2 = math.sqrt(sum(idf.get(f, 1.0) ** 2 for f in s2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def rule_similarity(r1, r2):
    """Similarite entre deux regles (moyenne des similarites L et R)."""
    sim_l = cosine_sim(r1.sL, r2.sL)
    sim_r = cosine_sim(r1.sR, r2.sR)
    return (sim_l + sim_r) / 2.0


def fuse_rules(r1, r2):
    """Fusionne deux regles en une seule (union des signatures)."""
    return Rule(
        sL=r1.sL | r2.sL,
        sR=r1.sR | r2.sR,
        rt=r1.rt,
        weight=r1.weight + r2.weight,
    )


class GRASPit:
    """Systeme de classification GRASP-it."""

    def __init__(self, extractor, fusion_threshold=FUSION_THRESHOLD,
                 mode="knn", k=5):
        """
        Args:
            extractor: SignatureExtractor
            fusion_threshold: seuil de similarite pour la fusion (mode fusion)
            mode: "fusion" (GRASP-it classique) ou "knn" (k plus proches voisins)
            k: nombre de voisins pour le mode knn
        """
        self.extractor = extractor
        self.fusion_threshold = fusion_threshold
        self.mode = mode
        self.k = k
        self.rules = []

    def train(self, train_data, progress=True):
        """Phase d'apprentissage."""
        rules_by_type = defaultdict(list)
        total = len(train_data)
        skipped = 0

        if progress:
            print(f"Apprentissage ({self.mode}) sur {total} exemples...")

        for i, (a, b, rt) in enumerate(train_data):
            set_a, set_b = self.extractor.extract_pair(a, b)

            if not set_a and not set_b:
                skipped += 1
                continue

            rule = Rule(sL=set_a, sR=set_b, rt=rt, weight=1.0)
            rules_by_type[rt].append(rule)

            if progress and (i + 1) % 500 == 0:
                print(f"  [{i+1}/{total}] Regles creees")

        if progress and skipped:
            print(f"  ({skipped} exemples ignores - signatures vides)")

        if self.mode == "fusion":
            self._train_fusion(rules_by_type, progress)
        else:
            # Mode knn : garder toutes les regles brutes
            self.rules = []
            for rules in rules_by_type.values():
                self.rules.extend(rules)
            if progress:
                print(f"  Total : {len(self.rules)} regles (mode knn, k={self.k})")

    def _train_fusion(self, rules_by_type, progress):
        """Entrainement avec fusion de regles (GRASP-it classique)."""
        if progress:
            print("\nFusion des regles...")

        self.rules = []
        for rt, rules in rules_by_type.items():
            fused_rules = self._fuse_rules_for_type(rules)
            self.rules.extend(fused_rules)
            if progress:
                print(f"  {rt}: {len(rules)} -> {len(fused_rules)} regles")

        if progress:
            print(f"\nTotal : {len(self.rules)} regles apres fusion")

    def _fuse_rules_for_type(self, rules):
        """Fusionne les regles d'un meme type par similarite."""
        if len(rules) <= 1:
            return rules

        changed = True
        while changed:
            changed = False
            new_rules = []
            used = [False] * len(rules)

            for i in range(len(rules)):
                if used[i]:
                    continue
                current = rules[i]
                for j in range(i + 1, len(rules)):
                    if used[j]:
                        continue
                    sim = rule_similarity(current, rules[j])
                    if sim >= self.fusion_threshold:
                        current = fuse_rules(current, rules[j])
                        used[j] = True
                        changed = True
                new_rules.append(current)

            rules = new_rules

        return rules

    def predict(self, a, b, top_n=1):
        """Predit la relation semantique entre A et B."""
        set_a, set_b = self.extractor.extract_pair(a, b)

        if self.mode == "knn":
            return self._predict_knn(set_a, set_b, top_n)
        return self._predict_best_match(set_a, set_b, top_n)

    def _predict_best_match(self, set_a, set_b, top_n):
        """Prediction par meilleur score (mode fusion)."""
        scores_by_type = defaultdict(list)

        for rule in self.rules:
            sim_l = cosine_sim(set_a, rule.sL)
            sim_r = cosine_sim(set_b, rule.sR)
            score = (sim_l + sim_r) / 2.0
            scores_by_type[rule.rt].append(score)

        type_scores = {}
        for rt, scores in scores_by_type.items():
            type_scores[rt] = max(scores)

        if not type_scores:
            return ("unknown", 0.0) if top_n == 1 else []

        ranked = sorted(type_scores.items(), key=lambda x: -x[1])
        if top_n == 1:
            return ranked[0]
        return ranked[:top_n]

    def _predict_knn(self, set_a, set_b, top_n):
        """
        Prediction par k plus proches voisins.

        Pour chaque regle, calculer la similarite avec l'entree.
        Prendre les k regles les plus similaires et voter.
        """
        scored_rules = []
        for rule in self.rules:
            sim_l = cosine_sim(set_a, rule.sL)
            sim_r = cosine_sim(set_b, rule.sR)
            score = (sim_l + sim_r) / 2.0
            scored_rules.append((score, rule.rt))

        # Trier par score decroissant
        scored_rules.sort(key=lambda x: -x[0])

        # Vote pondere par les k plus proches (score^2 pour ponderation plus forte)
        votes = defaultdict(float)
        for score, rt in scored_rules[:self.k]:
            votes[rt] += score * score

        if not votes:
            return ("unknown", 0.0) if top_n == 1 else []

        ranked = sorted(votes.items(), key=lambda x: -x[1])
        if top_n == 1:
            return ranked[0]
        return ranked[:top_n]

    def predict_explain(self, a, b, top_n=3):
        """Predit avec explication des scores."""
        set_a, set_b = self.extractor.extract_pair(a, b)

        scored_rules = []
        for rule in self.rules:
            sim_l = cosine_sim(set_a, rule.sL)
            sim_r = cosine_sim(set_b, rule.sR)
            score = (sim_l + sim_r) / 2.0
            scored_rules.append((score, rule.rt, sim_l, sim_r, rule.weight))

        scored_rules.sort(key=lambda x: -x[0])

        if self.mode == "knn":
            # Agreger les votes des k plus proches
            votes = defaultdict(lambda: {"score": 0.0, "count": 0, "best_sim_A": 0, "best_sim_B": 0})
            for score, rt, sim_l, sim_r, w in scored_rules[:self.k]:
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
        else:
            # Mode fusion : meilleur score par type
            best_by_type = {}
            for score, rt, sim_l, sim_r, w in scored_rules:
                if rt not in best_by_type or score > best_by_type[rt]["score"]:
                    best_by_type[rt] = {"score": score, "sim_A": sim_l, "sim_B": sim_r, "weight": w}
            type_scores = best_by_type

        ranked = sorted(type_scores.items(), key=lambda x: -x[1]["score"])
        return ranked[:top_n]


if __name__ == "__main__":
    from jdm_client import JDMClient

    client = JDMClient()
    extractor = SignatureExtractor(client)

    mini_train = [
        ("désert", "Algérie", "r_lieu"),
        ("tour", "Pise", "r_lieu"),
        ("tabouret", "bois", "r_objetmatiere"),
        ("cuillère", "bois", "r_objetmatiere"),
        ("saucisse", "Toulouse", "r_lieuorigine"),
        ("vin", "France", "r_lieuorigine"),
    ]

    for mode in ["fusion", "knn"]:
        print(f"\n{'='*50}")
        print(f"Mode : {mode}")
        print(f"{'='*50}")
        model = GRASPit(extractor, mode=mode, k=3)
        model.train(mini_train)

        test_cases = [("montagne", "Alpes"), ("chaise", "fer"), ("café", "Brésil")]
        for a, b in test_cases:
            result = model.predict_explain(a, b, top_n=3)
            print(f"\n'{a} de {b}' :")
            for rt, info in result:
                print(f"  {rt}: score={info['score']:.3f}")

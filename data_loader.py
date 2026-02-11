import json
import os
import random
from config import LEARN_DIR, RANDOM_SEED, TRAIN_RATIO

#Json apprentissage -> {A,B,Relation}
def load_corpus(learn_dir=LEARN_DIR):
    corpus = []
    for filename in sorted(os.listdir(learn_dir)):
        if not filename.endswith(".json"):
            continue
        relation_type = filename.replace(".json", "")
        filepath = os.path.join(learn_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            examples = json.load(f)
        for ex in examples:
            a = ex.get("A", "").strip()
            b = ex.get("B", "").strip()
            if a and b:
                corpus.append((a, b, relation_type))
    return corpus

# split 80/20 ( Inutile actuellement, fait juste perdre 20% de notre jeu de données, on l'utilisait pour la partie d'évaluation)
def split_train_test(corpus, train_ratio=TRAIN_RATIO, seed=RANDOM_SEED):
    rng = random.Random(seed)

    by_type = {}
    for a, b, rt in corpus:
        by_type.setdefault(rt, []).append((a, b, rt))

    train_set = []
    test_set = []

    for rt, examples in by_type.items():
        rng.shuffle(examples)
        split_idx = int(len(examples) * train_ratio)
        train_set.extend(examples[:split_idx])
        test_set.extend(examples[split_idx:])

    rng.shuffle(train_set)
    rng.shuffle(test_set)

    return train_set, test_set


def get_all_words(corpus):
    words = set()
    for a, b, _ in corpus:
        words.add(a)
        words.add(b)
    return words
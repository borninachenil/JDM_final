"""Chargement du corpus et split train/test."""

import json
import os
import random
import re

from config import LEARN_DIR, RANDOM_SEED, TRAIN_RATIO


def load_corpus(learn_dir=LEARN_DIR):
    """
    Charge tous les fichiers JSON du corpus.

    Retourne une liste de tuples (A, B, relation_type).
    """
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


def split_train_test(corpus, train_ratio=TRAIN_RATIO, seed=RANDOM_SEED):
    """
    Split stratifie : meme proportion de chaque relation dans train et test.

    Retourne (train_set, test_set).
    """
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


# Determinants francais pour detecter la definitude
_DET_DEF = {"le", "la", "les", "l", "du", "des", "au", "aux"}
_DET_INDEF = {"un", "une", "des"}

def detect_definiteness(b_text):
    """
    Detecte la presence d'un determinant dans le complement B.

    Retourne : "DEF", "INDEF", ou "NONE"
    """
    b_lower = b_text.lower().strip()

    # Verifier si B commence par un determinant
    # Ex: "la France" → DEF, "une femme" → INDEF, "bois" → NONE
    for det in sorted(_DET_DEF | _DET_INDEF, key=len, reverse=True):
        pattern = rf"^{re.escape(det)}[\s']"
        if re.match(pattern, b_lower):
            if det in _DET_DEF:
                return "DEF"
            return "INDEF"

    # Cas special : B commence par une majuscule (entite nommee) → souvent sans det
    if b_text and b_text[0].isupper():
        return "NONE"

    return "NONE"


def get_all_words(corpus):
    """Extrait l'ensemble unique de tous les mots A et B du corpus."""
    words = set()
    for a, b, _ in corpus:
        words.add(a)
        words.add(b)
    return words


if __name__ == "__main__":
    corpus = load_corpus()
    print(f"Corpus total : {len(corpus)} exemples")

    by_type = {}
    for a, b, rt in corpus:
        by_type.setdefault(rt, []).append((a, b))

    print("\nRepartition :")
    for rt in sorted(by_type.keys()):
        print(f"  {rt}: {len(by_type[rt])} exemples")

    train, test = split_train_test(corpus)
    print(f"\nTrain: {len(train)}, Test: {len(test)}")

    words = get_all_words(corpus)
    print(f"Mots uniques : {len(words)}")

    print("\nExemples de definitude :")
    samples = [
        "la France", "bois", "une femme", "Toulouse",
        "l'ouvrier", "fer", "des fleurs", "Van Gogh"
    ]
    for s in samples:
        print(f"  '{s}' → {detect_definiteness(s)}")

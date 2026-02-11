#!/usr/bin/env python3
"""Test rapide du pipeline avec un sous-ensemble du corpus."""

import time
import random

from config import LEARN_DIR, CACHE_DIR
from jdm_client import JDMClient
from signature import SignatureExtractor
from data_loader import load_corpus, split_train_test, get_all_words
from grasp_it import GRASPit
from evaluate import evaluate, confusion_matrix


def main():
    # Charger le corpus complet
    corpus = load_corpus(LEARN_DIR)

    # Prendre un echantillon plus large
    TRAIN_PER_TYPE = 500
    TEST_PER_TYPE = 100

    rng = random.Random(42)
    by_type = {}
    for a, b, rt in corpus:
        by_type.setdefault(rt, []).append((a, b, rt))

    train = []
    test = []
    for rt, examples in by_type.items():
        rng.shuffle(examples)
        train.extend(examples[:TRAIN_PER_TYPE])
        test.extend(examples[TRAIN_PER_TYPE:TRAIN_PER_TYPE + TEST_PER_TYPE])

    rng.shuffle(train)
    rng.shuffle(test)

    print(f"Mini-corpus : train={len(train)}, test={len(test)}")

    # Initialiser
    client = JDMClient(cache_dir=CACHE_DIR)
    extractor = SignatureExtractor(client)

    # Prefetch
    all_words = set()
    for a, b, _ in train + test:
        all_words.add(a)
        all_words.add(b)

    print(f"\nPrefetch de {len(all_words)} mots...")
    start = time.time()
    client.prefetch_batch(list(all_words), progress=True)
    print(f"Prefetch : {time.time() - start:.1f}s")

    # Test les deux modes
    for mode, k in [("knn", 5), ("knn", 7), ("knn", 11), ("knn", 15)]:
        print(f"\n{'='*60}")
        if mode == "knn":
            print(f"Mode : k-NN (k={k})")
            model = GRASPit(extractor, mode="knn", k=k)
        else:
            print(f"Mode : fusion")
            model = GRASPit(extractor, mode="fusion")
        print(f"{'='*60}")

        start = time.time()
        model.train(train, progress=False)
        print(f"Apprentissage : {time.time() - start:.1f}s ({len(model.rules)} regles)")

        print("Evaluation...")
        results = evaluate(model, test, progress=False)
        print(f"\n  >>> Macro F1 = {results['macro_f1']:.3f}, Accuracy = {results['accuracy']:.1%}")
        confusion_matrix(results["predictions"])


if __name__ == "__main__":
    main()

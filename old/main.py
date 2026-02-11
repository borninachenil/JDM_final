#!/usr/bin/env python3
"""
Projet "A de B" - Analyseur Semantique
HAI922 - TALN 2

Pipeline principal : chargement, prefetch JDM, apprentissage, evaluation, mode interactif.
"""
import sys
import time
import re
from config import LEARN_DIR, CACHE_DIR, RELATION_LABELS, TRAIN_RATIO
from jdm_client import JDMClient
from signature import SignatureExtractor
from data_loader import load_corpus, split_train_test, get_all_words
from grasp_it import GRASPit
from evaluate import evaluate, confusion_matrix


def parse_expression(expr):
    """
    Parse une expression 'A de B' en (A, B).

    Gere les variantes : "A de B", "A d'B", "A du B", "A des B".
    """
    expr = expr.strip()

    # Patterns de separation
    patterns = [
        r"^(.+?)\s+d[''](.+)$",      # A d'B
        r"^(.+?)\s+du\s+(.+)$",       # A du B
        r"^(.+?)\s+des\s+(.+)$",      # A des B
        r"^(.+?)\s+de\s+la\s+(.+)$",  # A de la B
        r"^(.+?)\s+de\s+l[''](.+)$",  # A de l'B
        r"^(.+?)\s+de\s+(.+)$",       # A de B (general)
    ]

    for pattern in patterns:
        match = re.match(pattern, expr, re.IGNORECASE)
        if match:
            a = match.group(1).strip()
            b = match.group(2).strip()
            return a, b
    return None, None


def interactive_mode(model):
    print("\n" + "=" * 60)
    print("MODE INTERACTIF - Entrez 'A de B' (ou 'q' pour quitter)")
    print("=" * 60)
    while True:
        try:
            expr = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        if expr.lower() in ("q", "quit", "exit", ""):
            print("Au revoir !")
            break

        a, b = parse_expression(expr)
        if a is None:
            print("Format invalide. Utilisez : 'A B'")
            continue

        print(f"  A = '{a}', B = '{b}'")

        start = time.time()
        results = model.predict_explain(a, b, top_n=5)
        elapsed = time.time() - start

        print(f"  Inference en {elapsed*1000:.0f}ms\n")
        print(f"  {'Rang':<6} {'Relation':<30} {'Score':>8}")
        print(f"  {'-'*50}")

        for i, (rt, info) in enumerate(results):
            label = RELATION_LABELS.get(rt, rt)
            marker = " <--" if i == 0 else ""
            print(f"  {i+1:<6} {label:<30} {info['score']:>8.3f}{marker}")


def main():
    print("=" * 60)
    print("PROJET 'A de B' - Analyseur Semantique")
    print("HAI922 - TALN 2")
    print("=" * 60)

    # Etape 1 : Charger le corpus
    print("\n[1/5] Chargement du corpus...")
    corpus = load_corpus(LEARN_DIR)
    train, test = split_train_test(corpus, train_ratio=TRAIN_RATIO)
    print(f"  Corpus : {len(corpus)} exemples")
    print(f"  Train  : {len(train)} exemples")
    print(f"  Test   : {len(test)} exemples")

    # Etape 2 : Initialiser le client JDM
    print("\n[2/5] Initialisation du client JDM...")
    client = JDMClient(cache_dir=CACHE_DIR)
    extractor = SignatureExtractor(client)

    # Etape 3 : Prefetch des donnees JDM
    print("\n[3/5] Prefetch des donnees JDM...")
    all_words = get_all_words(corpus)
    print(f"  {len(all_words)} mots uniques a charger")
    start = time.time()
    client.prefetch_batch(list(all_words), progress=True)
    prefetch_time = time.time() - start
    print(f"  Prefetch termine en {prefetch_time:.1f}s")

    # Etape 4 : Apprentissage
    print("\n[4/5] Apprentissage GRASP-it...")
    model = GRASPit(extractor)
    start = time.time()
    model.train(train)
    train_time = time.time() - start
    print(f"  Apprentissage termine en {train_time:.1f}s")

    # Etape 5 : Evaluation
    print("\n[5/5] Evaluation...")
    start = time.time()
    results = evaluate(model, test)
    eval_time = time.time() - start
    print(f"\n  Evaluation terminee en {eval_time:.1f}s")

    # Matrice de confusion
    confusion_matrix(results["predictions"])

    # Resume des temps
    print(f"\n{'='*60}")
    print(f"Resume des temps :")
    print(f"  Prefetch JDM : {prefetch_time:.1f}s")
    print(f"  Apprentissage : {train_time:.1f}s")
    print(f"  Evaluation ({len(test)} ex.) : {eval_time:.1f}s")
    print(f"  Temps inference moyen : {eval_time/len(test)*1000:.0f}ms")

    # Mode interactif
    if sys.stdin.isatty():
        interactive_mode(model)


if __name__ == "__main__":
    main()

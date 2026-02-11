import sys
import time
from config import LEARN_DIR, CACHE_DIR, RELATION_LABELS, TRAIN_RATIO
from jdm_client import JDMClient
from signature import SignatureExtractor
from data_loader import load_corpus, split_train_test, get_all_words
from grasp_it import GRASPit
from evaluate import evaluate, confusion_matrix
import re

def mot_connu(client, word):
    data = client.get_relations(word)
    return bool(data["nodes"])


def parse_expression(expr, client):
    expr = expr.strip()
    if not expr:
        return None, None

    # Trouver tous les points de coupure sur "de", "du", "des", "d'", "de la", "de l'"
    separators = re.finditer(
        r"\s+("
        r"de\s+la\s+|de\s+l['']\s*|"
        r"d['']\s*un\s+|d['']\s*une\s+|"
        r"d['']\s*|"
        r"du\s+|des\s+|"
        r"au\s+|aux\s+|"
        r"de\s+"
        r")", expr, re.IGNORECASE)
    splits = []
    for m in separators:
        a = expr[:m.start()].strip()
        b = expr[m.end():].strip()
        if a and b:
            splits.append((a, b))

    if not splits:
        return None, None

    # Cas simple
    if len(splits) == 1:
        return splits[0]

    # Cas ambigu
    for a, b in splits:
        if mot_connu(client, a) and mot_connu(client, b):
            return a, b

    return splits[0]


def interactive_mode(model, client):
    print("\n" + "=" * 60)
    print("MODE INTERACTIF - Entrez une expression (ou 'q' pour quitter)")
    print("=" * 60)
    while True:
        try:
            expr = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if expr.lower() in ("q", "quit", "exit", ""):
            break

        a, b = parse_expression(expr, client)
        if a is None or b is None:
            print("Aucun decoupage valide (termes non reconnus par JDM).")
            continue

        print(f"  A = '{a}', B = '{b}'")

        start = time.time()
        results = model.predict_2(a, b, top_n=5)
        elapsed = time.time() - start

        print(f"  Inference en {elapsed*1000:.0f}ms\n")
        print(f"  {'Rang':<6} {'Relation':<30} {'Score':>8}")
        print(f"  {'-'*50}")

        for i, (rt, info) in enumerate(results):
            label = RELATION_LABELS.get(rt, rt)
            marker = " <--" if i == 0 else ""
            print(f"  {i+1:<6} {label:<30} {info['score']:>8.3f}{marker}")


def main():
    # Corpus
    print("\n 1 Chargement du corpus...")
    corpus = load_corpus(LEARN_DIR)
    train, test = split_train_test(corpus, train_ratio=TRAIN_RATIO)
    print(f"  Corpus : {len(corpus)} exemples")
    print(f"  Train  : {len(train)} exemples")
    print(f"  Test   : {len(test)} exemples")

    #JDM
    print("\n 2 Initialisation du client JDM...")
    client = JDMClient(cache_dir=CACHE_DIR)
    extractor = SignatureExtractor(client)
    print("\n 3 Prefetch des donnees JDM...")
    all_words = get_all_words(corpus)
    print(f"  {len(all_words)} mots uniques a charger")
    start = time.time()
    client.prefetch_batch(list(all_words), progress=True)
    prefetch_time = time.time() - start
    print(f"  Prefetch termine en {prefetch_time:.1f}s")

    # Etape 4 : Apprentissage
    print("\n 4 Apprentissage GRASP-it...")
    model = GRASPit(extractor)
    start = time.time()
    model.train(train)
    train_time = time.time() - start
    print(f"  Apprentissage termine en {train_time:.1f}s")

    # actuellement cassé
    #Etape 5 : Evaluation
    #print("\n 5 Evaluation...")
    #start = time.time()
    #results = evaluate(model, test)
    #eval_time = time.time() - start
    #print(f"\n  Evaluation terminee en {eval_time:.1f}s")

    # Matrice de confusion
    #confusion_matrix(results["predictions"])

    # Resume des temps
    print(f"  Prefetch JDM : {prefetch_time:.1f}s")
    #print(f"  Apprentissage : {train_time:.1f}s")
    #print(f"  Evaluation ({len(test)} ex.) : {eval_time:.1f}s")
    #print(f"  Temps inference moyen : {eval_time/len(test)*1000:.0f}ms")

    # Mode interactif
    if sys.stdin.isatty():
        interactive_mode(model, client)


if __name__ == "__main__":
    main()

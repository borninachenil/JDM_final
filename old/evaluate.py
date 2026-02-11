"""Evaluation du systeme : precision, rappel, F1 par classe et global."""

from collections import defaultdict

from config import RELATION_LABELS


def evaluate(model, test_data, progress=True):
    """
    Evalue le modele sur les donnees de test.

    Retourne un dict avec les metriques par classe et globales.
    """
    predictions = []
    total = len(test_data)

    if progress:
        print(f"Evaluation sur {total} exemples...")

    for i, (a, b, rt_expected) in enumerate(test_data):
        rt_predicted, score = model.predict(a, b)
        predictions.append((rt_expected, rt_predicted))

        if progress and (i + 1) % 100 == 0:
            # Calculer l'accuracy courante
            correct = sum(1 for exp, pred in predictions if exp == pred)
            acc = correct / len(predictions)
            print(f"  [{i+1}/{total}] accuracy courante: {acc:.3f}")

    # Calcul des metriques par classe
    classes = sorted(set(rt for rt, _ in predictions) | set(rt for _, rt in predictions))

    # True positives, false positives, false negatives par classe
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for expected, predicted in predictions:
        if expected == predicted:
            tp[expected] += 1
        else:
            fp[predicted] += 1
            fn[expected] += 1

    # Metriques par classe
    results = {}
    for cls in classes:
        p = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
        r = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        support = tp[cls] + fn[cls]
        results[cls] = {"precision": p, "recall": r, "f1": f1, "support": support}

    # Moyennes macro
    n_classes = len(results)
    macro_p = sum(r["precision"] for r in results.values()) / n_classes if n_classes else 0
    macro_r = sum(r["recall"] for r in results.values()) / n_classes if n_classes else 0
    macro_f1 = sum(r["f1"] for r in results.values()) / n_classes if n_classes else 0

    # Accuracy globale
    correct = sum(1 for exp, pred in predictions if exp == pred)
    accuracy = correct / total if total > 0 else 0.0

    # Affichage
    print("\n" + "=" * 75)
    print(f"{'Type':<25} {'P':>8} {'R':>8} {'F1':>8} {'Support':>8}")
    print("-" * 75)

    for cls in sorted(results.keys()):
        r = results[cls]
        label = RELATION_LABELS.get(cls, cls)
        print(f"{label:<25} {r['precision']:>8.1%} {r['recall']:>8.1%} {r['f1']:>8.3f} {r['support']:>8d}")

    print("-" * 75)
    print(f"{'Macro avg':<25} {macro_p:>8.1%} {macro_r:>8.1%} {macro_f1:>8.3f} {total:>8d}")
    print(f"{'Accuracy':<25} {accuracy:>8.1%}")
    print("=" * 75)

    return {
        "per_class": results,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "predictions": predictions,
    }


def confusion_matrix(predictions, classes=None):
    """Affiche une matrice de confusion simplifiee (top erreurs)."""
    if classes is None:
        classes = sorted(set(rt for rt, _ in predictions) | set(rt for _, rt in predictions))

    errors = defaultdict(int)
    for expected, predicted in predictions:
        if expected != predicted:
            errors[(expected, predicted)] += 1

    if not errors:
        print("Aucune erreur !")
        return

    print("\nTop 15 confusions :")
    print(f"{'Attendu':<25} {'Predit':<25} {'Count':>6}")
    print("-" * 60)
    for (exp, pred), count in sorted(errors.items(), key=lambda x: -x[1])[:15]:
        exp_label = RELATION_LABELS.get(exp, exp)
        pred_label = RELATION_LABELS.get(pred, pred)
        print(f"{exp_label:<25} {pred_label:<25} {count:>6}")

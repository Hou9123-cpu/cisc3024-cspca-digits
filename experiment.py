"""Run the CPU-only handwritten digit recognition experiment."""

import argparse
import csv
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from cspca import fit_cspca


DIMENSIONS = (2, 4, 8, 16)
PRIMARY_DIMENSION = 4
KAPPAS = (0.1, 1.0, 10.0, 100.0)
SPLIT_SEED = 42
CV_SEED = 7


def predict(train_features, train_labels, query_features):
    classifier = KNeighborsClassifier(n_neighbors=3, algorithm="kd_tree", n_jobs=1)
    return classifier.fit(train_features, train_labels).predict(query_features)


def select_kappas(x, y):
    """Select by mean validation accuracy separately for each dimension."""
    scores = {q: {k: [] for k in KAPPAS} for q in DIMENSIONS}
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=CV_SEED)
    for train, val in folds.split(x, y):
        for kappa in KAPPAS:
            mean, axes = fit_cspca(x[train], y[train], max(DIMENSIONS), kappa)
            train_z = (x[train] - mean) @ axes
            val_z = (x[val] - mean) @ axes
            for q in DIMENSIONS:
                pred = predict(train_z[:, :q], y[train], val_z[:, :q])
                scores[q][kappa].append(accuracy_score(y[val], pred))
    # KAPPAS is ascending, so max picks the smaller kappa on an exact tie.
    chosen = {q: max(KAPPAS, key=lambda k: np.mean(scores[q][k])) for q in DIMENSIONS}
    return chosen, scores


def run(output_dir):
    start = perf_counter()
    x, y = load_digits(return_X_y=True)
    x = x / 16.0
    train, test = train_test_split(np.arange(len(y)), test_size=0.25,
                                   random_state=SPLIT_SEED, stratify=y)
    x_train, x_test = x[train], x[test]
    y_train, y_test = y[train], y[test]
    chosen, cv_scores = select_kappas(x_train, y_train)
    results = []
    predictions = {"index": test, "truth": y_test}

    def record(method, q, z_train, z_test, kappa=None):
        pred = predict(z_train, y_train, z_test)
        predictions[f"{method}_{q}"] = pred
        total_ss = np.sum((x_train - x_train.mean(axis=0)) ** 2)
        retained = np.sum((z_train - z_train.mean(axis=0)) ** 2) / total_ss
        results.append(dict(method=method, dimensions=q, kappa=kappa,
                            correct=int(np.sum(pred == y_test)),
                            accuracy=accuracy_score(y_test, pred),
                            macro_f1=f1_score(y_test, pred, average="macro"),
                            train_variance_fraction=float(retained)))

    pca = PCA(n_components=max(DIMENSIONS), svd_solver="full").fit(x_train)
    pca_train, pca_test = pca.transform(x_train), pca.transform(x_test)
    for q in DIMENSIONS:
        record("PCA", q, pca_train[:, :q], pca_test[:, :q])
        mean, axes = fit_cspca(x_train, y_train, q, chosen[q])
        record("CSPCA", q, (x_train - mean) @ axes, (x_test - mean) @ axes, chosen[q])
    record("Raw", x.shape[1], x_train, x_test)
    elapsed = perf_counter() - start

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(predictions)
        writer.writerows(zip(*predictions.values()))
    report = dict(train_size=len(train), test_size=len(test), split_seed=SPLIT_SEED,
                  cv_seed=CV_SEED, primary_dimension=PRIMARY_DIMENSION,
                  versions=dict(python=platform.python_version(), numpy=np.__version__,
                                scikit_learn=sklearn.__version__),
                  elapsed_seconds=elapsed, cv_fold_accuracies=cv_scores, results=results)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in results:
        print(f"{row['method']:5} {row['dimensions']:2}D  "
              f"{row['correct']:3}/{len(test)}  accuracy {row['accuracy']:.4f}")
    print(f"Experiment time: {elapsed:.3f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    with threadpool_limits(limits=1):
        run(parser.parse_args().output)

# Recognising handwritten digits with covariance-supervised PCA

Can training labels help a four-number representation distinguish handwritten digits? This CISC3024 experiment applies the class-kernel form of covariance-supervised PCA (CSPCA) to the 8 × 8 Digits images bundled with scikit-learn. Ordinary PCA and raw pixels are controls. Every representation uses the same Euclidean 3-nearest-neighbour classifier.

## Run locally

Tested on an Apple M2 with 8 GB of RAM, using Python 3.13.9. From this directory, run the following commands in a Python 3.13 environment. The dataset ships with scikit-learn and the experiment runs offline after setup.

```sh
python -m pip install -r requirements.txt
python -m unittest -v
VECLIB_MAXIMUM_THREADS=1 python experiment.py --output run
```

The experiment took 0.56 seconds on the recorded run, or 3.63 seconds including interpreter startup, imports and file output. Timing varies between runs.

## Experiment

1. Divide the 1,797 images by a stratified, shuffled 75/25 split (seed 42). Scale all pixel values by the known maximum of 16.
2. Within the 1,347 training images, use three stratified folds (seed 7) to select the covariance weight from 0.1, 1, 10 and 100. Select it independently for each dimension by mean validation accuracy, taking the smallest weight on a tie. Each fold fits its own pixel means and projection.
3. Refit on all training images and evaluate once per specified model on the same 450 held-out images. Four components are the primary comparison. Two, eight and sixteen components show sensitivity to compression.

| Components | PCA correct | CSPCA correct | PCA accuracy | CSPCA accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 270 | 280 | 60.00% | 62.22% |
| 4 | 388 | 394 | 86.22% | 87.56% |
| 8 | 433 | 437 | 96.22% | 97.11% |
| 16 | 441 | 442 | 98.00% | 98.22% |

Raw pixels (64 dimensions) classified 443/450 images correctly (98.44%). At four dimensions CSPCA retained 46.67% of training variance and PCA retained 48.95%. CSPCA corrected 18 PCA errors and introduced 12 on this split. The package is a subset of the UCI data, and the split is by image. These measurements do not estimate accuracy on unseen writers.

## Files

- `cspca.py` implements the class-kernel projection using class sums and a symmetric eigensolver.
- `test_cspca.py` checks the subspace against a direct implementation of the label kernel and tests invariance to pixel offsets and label names.
- `experiment.py` loads the images, tunes the projection, evaluates all representations and writes results.
- `results/metrics.json` records the original metrics, fold scores, package versions and compute time.
- `results/predictions.csv` lists the original scikit-learn sample index, true digit and each model's prediction for every held-out image.

## Sources

The projection follows Equation 15 in Papazoglou and Yin (2025), extended here from binary labels to the ten digit classes through the same equality kernel. The digit experiment, classifier and implementation were produced for this assignment using Codex. [Original preprint](https://arxiv.org/html/2506.19247v1)

The images come from the scikit-learn copy of the UCI optical recognition test subset. [Loader documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) · [UCI dataset](https://doi.org/10.24432/C50P49)

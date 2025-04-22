#!/usr/bin/env python3
"""
DP Private-PGM Synthetic Data Generator Script

This script reads X_train and y_train from text files, applies Private-PGM
to generate an (epsilon, delta)-DP synthetic dataset, and saves the outputs.

Usage:
    python dp_pgm_synth.py \
        --x_train_path path/to/X_train.txt \
        --y_train_path path/to/y_train.txt \
        --epsilon 1.0 \
        --delta 1e-6

Outputs are saved to `data/{epsilon}/{delta}/X_synth.txt` and
`data/{epsilon}/{delta}/y_synth.txt`.
"""
import os
import argparse
import numpy as np
import pandas as pd
import itertools
import time
from mbi import Domain
from mbi import LinearMeasurement
from mbi.estimation import mirror_descent

def dp_synthesize_private_pgm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epsilon: float,
    delta: float,
    max_marginal_order: int = 1,
    num_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate (epsilon, delta)-DP synthetic data via Private-PGM.
    Returns (X_synth, y_synth) as numpy arrays.
    """
    # 1) Combine features and label
    n, d = X_train.shape
    df = pd.DataFrame(X_train, columns=[f"X{i}" for i in range(d)])
    df["y"] = y_train.astype(int)

    # 2) Discretize and encode each column into integer categories
    cat_df = pd.DataFrame()
    names = []
    sizes = []
    for col in df.columns:
        binned = pd.qcut(df[col], q=num_bins, duplicates="drop").astype("category")
        cat_df[col] = binned.cat.codes
        names.append(col)
        sizes.append(len(binned.cat.categories))

    # 3) Define the discrete domain for PGM
    domain = Domain(names, sizes)

    # 4) Prepare all marginals up to the specified order
    attr_indices = list(range(len(names)))
    combinations = []
    for order in range(1, max_marginal_order + 1):
        combinations += list(itertools.combinations(attr_indices, order))

    # 5) Compute Gaussian noise scale (sensitivity = 1)
    sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    # 6) Measure each marginal privately
    measurements = []
    values = cat_df.values  # shape (n, d+1)
    for i, comb in enumerate(combinations):
        print(f"  Measuring marginal {i+1}/{len(combinations)}: {comb}")
        # comb: tuple of column indices
        comb_names = [names[i] for i in comb]
        # define histogram bins for each column
        bins = [np.arange(sizes[i] + 1) for i in comb]
        # true counts via histogramdd
        counts = np.histogramdd(values[:, comb].astype(int), bins=bins)[0].flatten()
        # add to measurements
        measurements.append(LinearMeasurement(counts, comb_names, sigma))

    # 7) Fit the maximum-entropy graphical model via mirror descent
    print("Fitting the maximum entropy graphical model via mirror descent")
    pgm = mirror_descent(
        domain,
        measurements
    )

    # 8) Sample synthetic records
    print(f"Sampling {n} records")
    synth_records = pgm.synthetic_data(n)
    synth_df = synth_records.df

    # 9) Extract synthetic arrays (integer codes)
    print("Extracting synthetic arrays")
    X_synth = np.stack(
        [synth_df[f"X{i}"].astype(int).values for i in range(d)],
        axis=1
    )
    y_synth = synth_df["y"].astype(int).values

    print("Done")

    return X_synth, y_synth


def main():
    parser = argparse.ArgumentParser(
        description="DP Private-PGM Synthetic Data Generator"
    )
    parser.add_argument(
        "--x_train_path", required=True,
        help="Path to X_train text file (numpy loadtxt format)"
    )
    parser.add_argument(
        "--y_train_path", required=True,
        help="Path to y_train text file (numpy loadtxt format)"
    )
    parser.add_argument(
        "--epsilon", type=float, required=True,
        help="Differential privacy epsilon"
    )
    parser.add_argument(
        "--delta", type=float, required=True,
        help="Differential privacy delta"
    )
    parser.add_argument(
        "--output_dir", default="data",
        help="Base directory for output (default: data)"
    )
    args = parser.parse_args()

    # Load input data
    X_train = np.loadtxt(args.x_train_path)
    y_train = np.loadtxt(args.y_train_path)

    start = time.time()

    # Generate synthetic data
    X_synth, y_synth = dp_synthesize_private_pgm(
        X_train, y_train,
        epsilon=args.epsilon,
        delta=args.delta
    )

    finish = time.time()

    # Prepare output directory: data/{epsilon}/{delta}
    out_dir = os.path.join(
        args.output_dir,
        str(args.epsilon),
        str(args.delta)
    )
    os.makedirs(out_dir, exist_ok=True)

    # Save synthetic arrays
    x_out_path = os.path.join(out_dir, "X_synth.txt")
    y_out_path = os.path.join(out_dir, "y_synth.txt")
    np.savetxt(x_out_path, X_synth, fmt='%d')
    np.savetxt(y_out_path, y_synth, fmt='%d')

    print(f"Saved X_synth to {x_out_path}")
    print(f"Saved y_synth to {y_out_path}")

    print(f"Finished in {finish - start}")

if __name__ == "__main__":
    main()
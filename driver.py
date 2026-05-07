import numpy as np
from sobol import SobolSampler
import os
import argparse
from setup import setup_simulation
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sobol Sampler Driver")
    parser.add_argument(
        "--gen_samples",
        action="store_true",
        help="Generate and save Sobol samples to CSV"
    )
    parser.add_argument(
        "--setup_nr",
        type=int,
        required=True,
        help="Number of simulation cases to set up"
    )
    args = parser.parse_args()

    current_dir = os.getcwd()
    csv_filename = os.path.join(current_dir, "sobol_samples.csv")

    if args.gen_samples:
        print(f"Generating samples and saving to {csv_filename}...")
        # Parameter limits for the Sobol sampler.
        # Reynolds and alpha are rounded to integers, so bounds include ±0.5 to give
        # endpoints equal probability. 'a' is sampled log-uniformly. 'n' is floored
        # to an index into n_choices.
        limits = {
            "re":    [999.5, 3900.5],
            "alpha": [-0.5, 45.5],
            "a":     [np.log10(1 / 6), np.log10(1.5)],
            "n":     [0, 4.0]
        }
        n_choices = [1, 2, 5, np.inf]
        tool = SobolSampler(limits, n_choices)
        data = tool.generate(2**10)
        tool.save(data, filename=csv_filename)
    else:
        print("Sample generation skipped. Use --gen_samples to generate.")

    if not os.path.exists(csv_filename):
        print(f"Error: {csv_filename} not found. Run with --gen_samples first.")
        quit()

    df = pd.read_csv(csv_filename)
    created_count = 0

    if args.setup_nr == 0:
        print("No cases have been set up.")
        quit()

    for index, row in df.iterrows():
        if created_count >= args.setup_nr:
            break

        # Assign train / val / test split based on sample index
        tag_nr = index % 10
        if tag_nr < 8:
            tag_cat = "train"
        elif tag_nr == 8:
            tag_cat = "val"
        else:
            tag_cat = "test"

        n_raw = row['n']
        my_vars = {
            "Re":           int(row['Reynolds']),
            "inflow_angle": row['alpha'] * np.pi / 180,
            "a":            round(row['a'], 3),
            "b":            0.5,
            "n":            np.inf if (str(n_raw).lower() == "inf" or np.isinf(n_raw)) else int(n_raw),
            "tag":          tag_cat,
            "GR_BL":        1.021,
            "t_BL":         0.5,
            "meshing":      True,
            "importing":    True,
            "partitioning": True,
            "initial_FTTs": 4,
            "FTT":          136,
            "overwrite":    False,
        }

        created_setup = setup_simulation(current_dir, my_vars)
        if created_setup:
            created_count += 1

    print(f"Done. Requested: {args.setup_nr} | Successfully created: {created_count}")

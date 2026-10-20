"""Loading the substrate data and the precomputed feature sets."""

import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(ROOT, "cof_matched", "data_substrate_cofs.csv")
FEATURES_DIR = os.path.join(ROOT, "cof_matched", "rth_0.99", "features")
RESULTS_DIR = os.path.join(ROOT, "regressors_experiments", "results")
TARGET = "capacity_max"

# Short names for the precomputed single feature sets (from precursor2cof-capacitance).
FEATURE_SETS = {
    "descriptor": "features_rth_0.99_descriptor_False_False.csv",
    "custom_pattern": "features_rth_0.99_custom_pattern_True.csv",
    "ecfp": "features_rth_0.99_ecfp_2_1024_True.csv",
    "estate": "features_rth_0.99_estate_sum.csv",
    "functional_groups": "features_rth_0.99_functional_groups_True.csv",
    "maccs": "features_rth_0.99_maccs_True.csv",
}


def load_data() -> pd.DataFrame:
    """SMILES and target, in the same row order as every feature file."""
    return pd.read_csv(DATA_PATH)


def load_features(*names: str) -> pd.DataFrame:
    """Concatenate the named feature sets column-wise (without smiles and target)."""
    parts = []
    for name in names:
        df = pd.read_csv(os.path.join(FEATURES_DIR, FEATURE_SETS[name]))
        parts.append(df.drop(columns=["smiles", TARGET]))
    X = pd.concat(parts, axis=1)
    return X.loc[:, ~X.columns.duplicated()]


def readable_names(columns) -> list[str]:
    """Human-readable names for precomputed columns.

    `functional_groups_i` is RDKit's i-th `fr_*` fragment count and `estate_i` the EState sum of the
    i-th RDKit EState atom type (both checked against the precomputed values). Other names are kept.
    """
    from rdkit.Chem import Descriptors
    from rdkit.Chem.EState import AtomTypes

    fragments = [n for n, _ in Descriptors.descList if n.startswith("fr_")]
    estate_types = [n for n, _ in AtomTypes._rawD]
    out = []
    for c in columns:
        prefix, _, idx = c.rpartition("_")
        if prefix == "functional_groups" and idx.isdigit():
            out.append(fragments[int(idx)])
        elif prefix == "estate" and idx.isdigit():
            out.append(f"estate_{estate_types[int(idx)]}")
        else:
            out.append(c)
    return out

"""New features computed from SMILES (not in the precomputed sets)."""

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# Faraday constant / 3.6 C per mAh: theoretical capacity (mAh/g) = F * n_electrons / (3.6 * MW).
FARADAY_MAH = 96485.33212 / 3.6

# Groups commonly treated as redox-active in COF/organic electrodes, one electron per match.
REDOX_GROUPS = {
    "carbonyl": "[CX3]=[OX1]",
    "imine": "[CX3]=[NX2;!R]",
    "aromatic_n": "[nX2]",
    "ring_c_n_double": "[C;R]=[N;R]",
    "nitrile": "[CX2]#[NX1]",
    "azo": "[NX2]=[NX2]",
    "nitro": "[N+](=O)[O-]",
}
_REDOX = {k: Chem.MolFromSmarts(v) for k, v in REDOX_GROUPS.items()}


def redox_features(smiles: list[str]) -> pd.DataFrame:
    """Counts of redox-active groups, their density per mass and a theoretical-capacity estimate."""
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        mw = Descriptors.MolWt(mol)
        counts = {f"n_{k}": len(mol.GetSubstructMatches(p)) for k, p in _REDOX.items()}
        n_sites = sum(counts.values())
        heavy = mol.GetNumHeavyAtoms()
        row = dict(counts)
        row["n_redox_sites"] = n_sites
        row["theoretical_capacity"] = FARADAY_MAH * n_sites / mw
        row["heteroatom_capacity"] = FARADAY_MAH * rdMolDescriptors.CalcNumHeteroatoms(mol) / mw
        row["redox_sites_per_heavy_atom"] = n_sites / heavy
        row["log_mw"] = np.log(mw)
        rows.append(row)
    return pd.DataFrame(rows)


def rdkit_descriptors(smiles: list[str]) -> pd.DataFrame:
    """All RDKit 2D descriptors; constant and non-finite columns are dropped."""
    names = [n for n, _ in Descriptors.descList]
    rows = [Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(s)) for s in smiles]
    df = pd.DataFrame(rows, columns=names).replace([np.inf, -np.inf], np.nan)
    df = df.loc[:, df.notna().all() & (df.nunique() > 1)]
    return df.astype(float)


def count_fingerprint(smiles: list[str], kind: str = "morgan", radius: int = 2, size: int = 2048) -> pd.DataFrame:
    """Unfolded-collision count fingerprints (Morgan / atom pair / topological torsion)."""
    gen = {
        "morgan": lambda: rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=size),
        "fmorgan": lambda: rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=size, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        ),
        "atompair": lambda: rdFingerprintGenerator.GetAtomPairGenerator(fpSize=size),
        "torsion": lambda: rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=size),
    }[kind]()
    fps = np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=float)
    df = pd.DataFrame(fps, columns=[f"{kind}{radius if 'morgan' in kind else ''}_{i}" for i in range(size)])
    return df.loc[:, df.sum() > 0]

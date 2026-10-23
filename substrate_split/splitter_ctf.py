"""CTF-only main/side chain split -> output/ctf_split.csv.

CTF filter is `extract_ctfs` from precursor2cof-capacitance/notebooks/initial_data_processing.ipynb
(any C#N in the molecule). The split itself is splitter.split_substrate.
"""

import os

import pandas as pd
from rdkit import Chem

import splitter
from splitter import split_substrate

splitter.TIER1_END_GROUPS = [("nitrile", "[CX2]#[NX1]")]
splitter.TIER2_END_GROUPS = []

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "cof_matched", "data_substrate_cofs.csv")
OUT = os.path.join(HERE, "output", "ctf_split.csv")

# Tetra-nitrile pyrrolopyrrole: with 4 C#N ends our splitter makes the whole molecule main chain.
# Correct split is the hardcoded core from scripts/instance_importance_analysis_core_side.py
# (pyrrolopyrrole + 2 N-aryl nitrile arms), with the 2 C-aryl benzonitriles as side chains.
PYRROLOPYRROLE_CORE = Chem.MolFromSmarts(
    "N#Cc(cc1)ccc1-[n](cc1)c(cc2)c1[n]2-c(cc1)ccc1C#N"
)
CORE_OVERRIDE = {
    "N#Cc1ccc(-c2cc3c(cc(-c4ccc(C#N)cc4)n3-c3ccc(C#N)cc3)n2-c2ccc(C#N)cc2)cc1",
}


def core_override(smiles):
    mol = Chem.MolFromSmiles(smiles)
    main = set(mol.GetSubstructMatch(PYRROLOPYRROLE_CORE))
    side_chains = splitter._components(mol, set(range(mol.GetNumAtoms())) - main)
    main_smi, side_smis = splitter._fragment_smiles(mol, main, side_chains)
    return {
        "main_chain_smiles": main_smi,
        "side_chain_smiles": ".".join(side_smis),
        "note": "overridden: main chain = hardcoded pyrrolopyrrole core",
    }


def extract_ctfs(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        print("Invalid SMILES string.")
        return False
    nitrile_pattern = Chem.MolFromSmarts("[C]#[N]")
    return mol.HasSubstructMatch(nitrile_pattern)


data = pd.read_csv(DATA)
data["idx"] = range(len(data))
ctfs = data[data["smiles"].apply(extract_ctfs)]

rows = []
for idx, smi, cap in zip(ctfs.idx, ctfs.smiles, ctfs.capacity_max):
    s = split_substrate(smi)
    row = {
        "idx": idx,
        "smiles": smi,
        "capacity_max": cap,
        "n_ends": s.n_ends,
        "main_chain_smiles": s.main_chain_smiles,
        "side_chain_smiles": ".".join(s.side_chain_smiles),
        "note": "",
    }
    if smi in CORE_OVERRIDE:
        row.update(core_override(smi))
    row["more_than_2_ends"] = s.n_ends > 2
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"{len(df)} CTFs -> {OUT}")
print(df.loc[df.more_than_2_ends, ["idx", "n_ends", "smiles", "note"]].to_string())

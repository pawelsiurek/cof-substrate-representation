"""Frozen embeddings from pretrained molecular models (needs the `gpu` extra)."""

import os

import numpy as np
import pandas as pd

PRETRAINED_DIR = os.path.join(os.path.dirname(__file__), "..", "pretrained")
CHEMELEON_PATH = os.path.join(PRETRAINED_DIR, "chemeleon_mp.pt")


def chemeleon_embeddings(smiles: list[str], batch_size: int = 64) -> pd.DataFrame:
    """Mean-aggregated CheMeleon message-passing embeddings (loaded as in precursor2cof-capacitance)."""
    import torch
    from chemprop import data, featurizers, nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CHEMELEON_PATH, weights_only=True)
    mp = nn.BondMessagePassing(**ckpt["hyper_parameters"])
    mp.load_state_dict(ckpt["state_dict"])
    mp.eval().to(device)
    agg = nn.MeanAggregation()

    dset = data.MoleculeDataset([data.MoleculeDatapoint.from_smi(s) for s in smiles], featurizers.SimpleMoleculeMolGraphFeaturizer())
    loader = data.build_dataloader(dset, batch_size=batch_size, shuffle=False, num_workers=0)
    out = []
    with torch.no_grad():
        for batch in loader:
            bmg = batch.bmg
            bmg.to(device)
            out.append(agg(mp(bmg), bmg.batch).cpu().numpy())
    emb = np.vstack(out)
    return pd.DataFrame(emb, columns=[f"chemeleon_{i}" for i in range(emb.shape[1])])


def cached_chemeleon(smiles: list[str]) -> pd.DataFrame:
    """CheMeleon embeddings cached in results/emb_chemeleon.csv (rows in data order)."""
    from cofreg.data import RESULTS_DIR

    path = os.path.join(RESULTS_DIR, "emb_chemeleon.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    emb = chemeleon_embeddings(smiles)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    emb.to_csv(path, index=False)
    return emb


class ChemeleonFinetune:
    """Fine-tune CheMeleon message passing + a small FFN head (chemprop) on the target.

    X must be a single column of row indices into `smiles` (cross_val_oof only passes numeric arrays).
    Trains for a fixed number of epochs without a validation split, so all training data is used.
    """

    def __init__(self, smiles: tuple = (), epochs: int = 50, batch_size: int = 32, lr: float | None = None, seed: int = 0):
        self.smiles = smiles
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

    def get_params(self, deep=True):
        return {"smiles": self.smiles, "epochs": self.epochs, "batch_size": self.batch_size, "lr": self.lr, "seed": self.seed}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _dataset(self, X, y=None):
        from chemprop import data, featurizers

        idx = np.asarray(X).ravel().astype(int)
        ys = [None] * len(idx) if y is None else [np.array([v], dtype=float) for v in y]
        points = [data.MoleculeDatapoint.from_smi(self.smiles[i], t) for i, t in zip(idx, ys)]
        return data.MoleculeDataset(points, featurizers.SimpleMoleculeMolGraphFeaturizer())

    def _trainer(self):
        import logging

        import torch
        from lightning import pytorch as pl

        for name in ["lightning.pytorch.utilities.rank_zero", "lightning.pytorch.accelerators.cuda", "lightning.fabric"]:
            logging.getLogger(name).setLevel(logging.ERROR)

        return pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=self.epochs,
        )

    def fit(self, X, y):
        import torch
        from chemprop import data, models, nn
        from lightning import pytorch as pl

        pl.seed_everything(self.seed, verbose=False)
        dset = self._dataset(X, np.asarray(y, dtype=float))
        scaler = dset.normalize_targets()
        ckpt = torch.load(CHEMELEON_PATH, weights_only=True)
        mp = nn.BondMessagePassing(**ckpt["hyper_parameters"])
        mp.load_state_dict(ckpt["state_dict"])
        ffn = nn.RegressionFFN(input_dim=mp.output_dim, output_transform=nn.UnscaleTransform.from_standard_scaler(scaler))
        kwargs = {} if self.lr is None else {"init_lr": self.lr / 10, "max_lr": self.lr, "final_lr": self.lr / 10}
        self.model_ = models.MPNN(mp, nn.MeanAggregation(), ffn, batch_norm=False, **kwargs)
        loader = data.build_dataloader(dset, batch_size=self.batch_size, num_workers=0, seed=self.seed)
        self._trainer().fit(self.model_, loader)
        return self

    def predict(self, X):
        import torch
        from chemprop import data

        loader = data.build_dataloader(self._dataset(X), batch_size=64, shuffle=False, num_workers=0)
        preds = self._trainer().predict(self.model_, loader)
        return torch.cat(preds).numpy().ravel()

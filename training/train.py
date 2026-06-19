import copy
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


def apply_reproducibility():
    env_seed = os.getenv("TRAINING_SEED", "42")
    seed = int(env_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    print(f"Seed set to {seed} from environment variable.")
    return seed


class _SplitDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


def _split_dataframe(df: pd.DataFrame, label_column, train_ratio, val_ratio, seed):
    if label_column and label_column in df.columns:
        y = df[label_column].values
        X_df = df.drop(columns=[label_column])
    else:
        X_df = df.iloc[:, :-1]
        y = df.iloc[:, -1].values

    n_bins = max(2, min(10, len(df) // 10))
    try:
        strata = pd.qcut(y, q=n_bins, labels=False, duplicates="drop").astype(int)
    except Exception:
        strata = np.zeros(len(y), dtype=int)

    rng = np.random.default_rng(seed)

    i_train, i_val, i_test = [], [], []
    for bin_id in np.unique(strata):
        bin_idx = np.where(strata == bin_id)[0]
        bin_idx = rng.permutation(bin_idx)
        n_b = len(bin_idx)
        n_tr = max(1, int(n_b * train_ratio))
        n_va = max(1, int(n_b * val_ratio))
        i_train.extend(bin_idx[:n_tr].tolist())
        i_val.extend(bin_idx[n_tr:n_tr + n_va].tolist())
        i_test.extend(bin_idx[n_tr + n_va:].tolist())

    i_train = np.array(i_train)
    i_val   = np.array(i_val)
    i_test  = np.array(i_test)


    label_col = label_column if (label_column and label_column in df.columns) else df.columns[-1]
    feature_cols = [c for c in df.columns if c != label_col]

    def make_df(indices):
        out = X_df.iloc[indices].copy()
        out[label_col] = y[indices]
        out.columns = feature_cols + [label_col]
        return out.reset_index(drop=True)

    split_dfs = {
        "train": make_df(i_train),
        "val":   make_df(i_val),
        "test":  make_df(i_test),
    }

    X = X_df.values

    scaler = MinMaxScaler()
    X_train_norm = np.nan_to_num(scaler.fit_transform(X[i_train]))
    X_val_norm   = np.nan_to_num(scaler.transform(X[i_val]))
    X_test_norm  = np.nan_to_num(scaler.transform(X[i_test]))

    for split_name, x_norm, indices in [
        ("train", X_train_norm, i_train),
        ("val",   X_val_norm,   i_val),
        ("test",  X_test_norm,  i_test),
    ]:
        split_dfs[split_name][feature_cols] = x_norm

    return (
        _SplitDataset(X_train_norm, y[i_train]),
        _SplitDataset(X_val_norm,   y[i_val]),
        _SplitDataset(X_test_norm,  y[i_test]),
        split_dfs,
        scaler,
    )


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mse  = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2   = float(1 - np.sum((y_true - y_pred) ** 2) / ss_tot) if ss_tot > 0 else 0.0
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def train(
    model,
    csv_path,
    patience=10,
    max_epochs=500,
    label_column=None,
    train_ratio=0.70,
    val_ratio=0.15,
    min_delta=0.0,
    lr=1e-3,
    progress_callback=None,
):
    """
    Trains model with early stopping on validation MSE.
    Splits the dataset into train / val / test.

    Returns:
        model       — best checkpoint restored
        metrics     — dict with MAE, MSE, RMSE, R² evaluated on the test set
        split_dfs   — dict {"train": df, "val": df, "test": df}
    """
    seed = apply_reproducibility()

    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0 to leave room for a test set")

    df = pd.read_csv(csv_path)
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    if n_dropped:
        msg = f"Dropped {n_dropped} rows with NaN values ({n_before} → {len(df)} rows)"
        print(msg)
        if progress_callback:
            progress_callback({"type": "log", "message": msg})

    train_ds, val_ds, test_ds, split_dfs, scaler = _split_dataframe(
        df, label_column, train_ratio, val_ratio, seed
    )

    split_msg = f"Split → train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)} samples"
    print(split_msg)
    if progress_callback:
        progress_callback({"type": "log", "message": split_msg})

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=32)
    test_loader  = DataLoader(test_ds,  batch_size=32)

    device = torch.device("cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = copy.deepcopy(model.state_dict())
    no_improve    = 0

    setup_msg = f"patience={patience} | max_epochs={max_epochs} | lr={lr}"
    print(setup_msg)
    if progress_callback:
        progress_callback({"type": "log", "message": setup_msg})

    for epoch in range(max_epochs):
        # --- train ---
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x.unsqueeze(2)).squeeze(-1).squeeze(-1)
            loss = criterion(out, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # --- val ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vx, vy in val_loader:
                v_out = model(vx.unsqueeze(2)).squeeze(-1).squeeze(-1)
                val_loss += criterion(v_out, vy).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)

        if avg_val < best_val_loss - min_delta:
            best_val_loss = avg_val
            best_state    = copy.deepcopy(model.state_dict())
            no_improve    = 0
        else:
            no_improve += 1

        print(
            f"Epoch {epoch+1}/{max_epochs} | "
            f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
            f"no_improve: {no_improve}/{patience}"
        )
        if progress_callback:
            progress_callback({
                "type": "epoch",
                "epoch": epoch + 1,
                "max_epochs": max_epochs,
                "train_loss": round(avg_train, 6),
                "val_loss": round(avg_val, 6),
                "no_improve": no_improve,
                "patience": patience,
            })

        if no_improve >= patience:
            msg = f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}"
            print(msg)
            if progress_callback:
                progress_callback({"type": "log", "message": msg})
            break

    # Restore best checkpoint
    model.load_state_dict(best_state)

    # --- test evaluation ---
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for tx, ty in test_loader:
            preds = model(tx.unsqueeze(2)).squeeze(-1).squeeze(-1)
            all_preds.extend(preds.numpy())
            all_targets.extend(ty.numpy())

    metrics = _compute_metrics(np.array(all_targets), np.array(all_preds))
    test_msg = (
        f"Test → MAE: {metrics['mae']:.4f} | "
        f"RMSE: {metrics['rmse']:.4f} | R²: {metrics['r2']:.4f}"
    )
    print(test_msg)
    if progress_callback:
        progress_callback({"type": "log", "message": test_msg})

    return model, metrics, split_dfs

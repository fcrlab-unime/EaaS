import os
import torch
import numpy as np
import random

def apply_reproducibility():
    env_seed = os.getenv("TRAINING_SEED", "42")
    seed = int(env_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    print(f"Seed impostato a {seed} da variabile d'ambiente.")


import pandas as pd
from torch.utils.data import Dataset, DataLoader

class TrainCSVDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = df.iloc[:, :-1].values.astype(np.float32)
        self.y = df.iloc[:, -1].values.astype(np.float32)  # float32 per regressione

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


def train(model, csv_path, epochs, lr=1e-3):
    apply_reproducibility()
    device = torch.device("cpu")
    model.to(device)

    full_train_dataset = TrainCSVDataset(csv_path)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    print(f"Starting training: {len(train_data)} samples for training, {len(val_data)} for validation.")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            batch_x = batch_x.unsqueeze(2)  # [B, F] → [B, F, 1]
            outputs = model(batch_x).squeeze(-1).squeeze(-1)  # → [B]
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.unsqueeze(2)
                v_out = model(vx).squeeze(-1).squeeze(-1)
                v_loss = criterion(v_out, vy)
                val_loss += v_loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

    return model
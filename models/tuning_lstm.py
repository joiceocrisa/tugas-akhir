import torch
import torch.nn as nn
from models import LSTMModel
from torch.utils.data import DataLoader

def train_lstm_tuning(dropout, lr, batch_size, train_dataset, val_dataset, DEVICE):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=1,
        dropout=dropout
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    EPOCHS_TUNING = 20

    # ===================== TRAIN =====================
    for epoch in range(EPOCHS_TUNING):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # ===================== VALIDATION =====================
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)

    val_loss /= len(val_loader.dataset)
    return val_loss

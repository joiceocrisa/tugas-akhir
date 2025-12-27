import copy
import math
import torch
import torch.nn as nn
from models import FEDformer_Model
from torch.utils.data import DataLoader


def train_fedformer_tuning(
    dropout,
    lr,
    batch_size,
    train_dataset,
    val_dataset,
    DEVICE,
    base_configs
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    configs = copy.deepcopy(base_configs)
    configs.dropout = dropout

    model = FEDformer_Model(configs).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    EPOCHS_TUNING = 15

    # ===================== TRAIN =====================
    model.train()
    for epoch in range(EPOCHS_TUNING):
        for x_enc, y_true in train_loader:
            x_enc = x_enc.to(DEVICE)
            y_true = y_true.to(DEVICE)

            B, _, C = x_enc.shape

            x_dec = torch.zeros(
                B,
                configs.label_len + configs.pred_len,
                C,
                device=DEVICE
            )

            x_mark_enc = torch.zeros(
                B, configs.seq_len, 4, device=DEVICE
            )

            x_mark_dec = torch.zeros(
                B,
                configs.label_len + configs.pred_len,
                4, device=DEVICE
            )

            optimizer.zero_grad()

            outputs = model(
                x_enc,
                x_mark_enc,
                x_dec,
                x_mark_dec
            )

            if y_true.dim() == 2:
                y_true = y_true.unsqueeze(-1)

            loss = criterion(outputs, y_true)

            # NaN guard 
            if torch.isnan(loss):
                return float('inf')

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # ===================== VALIDATION =====================
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x_enc, y_true in val_loader:
            x_enc = x_enc.to(DEVICE)
            y_true = y_true.to(DEVICE)

            B, _, C = x_enc.shape

            x_dec = torch.zeros(
                B,
                configs.label_len + configs.pred_len,
                C,
                device=DEVICE
            )

            x_mark_enc = torch.zeros(
                B, configs.seq_len, 4, device=DEVICE
            )

            x_mark_dec = torch.zeros(
                B,
                configs.label_len + configs.pred_len,
                4, device=DEVICE
            )

            outputs = model(
                x_enc,
                x_mark_enc,
                x_dec,
                x_mark_dec
            )

            if y_true.dim() == 2:
                y_true = y_true.unsqueeze(-1)

            loss = criterion(outputs, y_true)

            if torch.isnan(loss):
                return float('inf')

            val_loss += loss.item() * x_enc.size(0)

    val_loss /= len(val_loader.dataset)

    # Safety check akhir
    if math.isnan(val_loss) or math.isinf(val_loss):
        return float('inf')

    return val_loss
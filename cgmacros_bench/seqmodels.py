"""LSTM and Transformer sequence models used in the paper's secondary experiment (CPU, small)."""
import numpy as np
import torch
import torch.nn as nn

SEQ_STEPS, SEQ_GAP = 12, 10          # 12 steps x 10 minutes = 110-minute window
PATIENCE = 3
SEQ_COLS = ["gl_now", "gl_d15", "min_since_reading", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "heart_rate",
            "cal_activity", "mets", "steps", "intensity", "time_since_meal", "carbs_2h", "cal_2h",
            "last_m_cal", "last_m_carbs", "last_m_protein", "last_m_fat", "last_m_fiber", "last_m_sugar",
            "last_m_amount"]
# two configurations per model; the one with the lower validation MSE is used on the test fold
SEQ_CONFIGS = [dict(hid=64, drop=0.2, lr=2e-3, wd=1e-4, noise=0.0),
               dict(hid=32, drop=0.4, lr=5e-4, wd=1e-3, noise=0.1)]


class SeqLSTM(nn.Module):
    def __init__(self, k, hid=64, drop=0.2):
        super().__init__()
        self.rnn = nn.LSTM(k, hid, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(hid, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(o[:, -1])


class SeqTF(nn.Module):
    def __init__(self, k, hid=64, drop=0.2, steps=SEQ_STEPS):
        super().__init__()
        d = hid
        self.proj = nn.Linear(k, d)
        self.pos = nn.Parameter(torch.zeros(1, steps, d))
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, drop, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        z = self.enc(self.proj(x) + self.pos)
        return self.head(z.mean(1))


def train_seq(model, Xtr, Ytr, Xva, Yva, epochs, lr=2e-3, wd=1e-4, noise=0.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best, bstate, wait = 1e9, None, 0
    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            xb = Xtr[b]
            if noise:
                xb = xb + noise * torch.randn_like(xb)
            loss = nn.functional.mse_loss(model(xb), Ytr[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([nn.functional.mse_loss(model(Xva[i:i + 4096]), Yva[i:i + 4096]).item()
                                for i in range(0, len(Xva), 4096)]))
        if vl < best - 1e-4:
            best, wait = vl, 0
            bstate = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(bstate)
    return model, ep + 1, best


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + 4096]) for i in range(0, len(X), 4096)]).numpy()

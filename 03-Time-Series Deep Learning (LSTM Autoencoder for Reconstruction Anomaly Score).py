import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# 1) Build sliding windows
def to_windows(series: np.ndarray, win: int):
    # series shape: (T, D) or (T,) -> we convert to (T, D)
    series = series if series.ndim == 2 else series[:, None]
    X = np.stack([series[i:i+win] for i in range(len(series)-win+1)], axis=0)
    return X  

# Example synthetic time-series
rng = np.random.RandomState(0)
T = 5000
x = np.sin(np.linspace(0, 60, T)) + 0.05*rng.randn(T)  # base
spike_idx = rng.choice(np.arange(200, T-200), size=20, replace=False)
x[spike_idx] += rng.uniform(4, 7, size=20)             # anomalies
X = x.reshape(-1, 1)

train_T = 2000
scaler = StandardScaler().fit(X[:train_T])
X_s = scaler.transform(X)

win = 64
X_train = to_windows(X_s[:train_T], win)   
X_all   = to_windows(X_s, win)             

# 2) LSTM Autoencoder 
class LSTMAE(nn.Module):
    def __init__(self, in_dim=1, hid=32, z=16, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(input_size=in_dim, hidden_size=hid, num_layers=num_layers, batch_first=True)
        self.to_z = nn.Linear(hid, z)
        self.from_z = nn.Linear(z, hid)
        self.decoder = nn.LSTM(input_size=hid, hidden_size=in_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        # x: (B, T, D)
        enc_out, _ = self.encoder(x)            # (B, T, hid)
        h_T = enc_out[:, -1, :]                 # last hidden state
        z = self.to_z(h_T)                      # (B, z)
        h0 = torch.tanh(self.from_z(z))         
        H = h0.unsqueeze(1).repeat(1, x.size(1), 1)  # (B, T, hid)
        dec_out, _ = self.decoder(H)            # (B, T, D)
        return dec_out

# 3) Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMAE(in_dim=1, hid=48, z=24).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.MSELoss()

train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)

model.train()
for epoch in range(15):
    loss_sum = 0.0
    for (batch,) in train_dl:
        batch = batch.to(device)
        opt.zero_grad()
        recon = model(batch)
        loss = crit(recon, batch)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * len(batch)
    print(f"epoch {epoch+1:02d} | train recon MSE: {loss_sum/len(train_ds):.6f}")

# 4) Anomaly scoring 
model.eval()
with torch.no_grad():
    X_all_t = torch.tensor(X_all, dtype=torch.float32, device=device)
    recon = model(X_all_t).cpu().numpy()
    err = np.mean((recon - X_all)**2, axis=(1, 2))  # per-window MSE

# Map window score back to time index
time_idx = np.arange(win-1, win-1 + len(err))
# Choose threshold: percentile or mean+K*std
thr = np.percentile(err, 95)  
anom_idx = time_idx[err >= thr]
print("\n")
print("03-Time-Series Deep Learning (LSTM Autoencoder for Reconstruction Anomaly Score)")
print("Detected anomalous time indices:", anom_idx[:50])
print("\n")
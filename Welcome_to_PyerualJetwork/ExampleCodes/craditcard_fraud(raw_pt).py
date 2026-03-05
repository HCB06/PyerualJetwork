import pandas as pd
import numpy as np
from pyerualjetwork.cpu import data_ops
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cpu")
print(f"Kullanılan cihaz : {device}\n")

# dataset link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
data = pd.read_csv('creditcard.csv')

x = data.drop('Class', axis=1).values
y = data['Class'].values

x = data_ops.normalization(x)

x_train, x_test, y_train, y_test = data_ops.split(x, y, 0.4, 42)

y_train, y_test = data_ops.encode_one_hot(y_train, y_test)

x_train, y_train = data_ops.synthetic_augmentation(x_train, y_train)

num_classes = y_train.shape[1]
input_size  = x_train.shape[1]

print(f"Eğitim seti : {x_train.shape}")
print(f"Test seti   : {x_test.shape}\n")

# ── Tensor'lara çevir ─────────────────────────────────────────────────────────
x_train_t = torch.tensor(x_train, dtype=torch.float32)
x_test_t  = torch.tensor(x_test,  dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32)

batch_size   = 32
train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(TensorDataset(x_test_t,  y_test_t),  batch_size=batch_size, shuffle=False)

# ── Model ─────────────────────────────────────────────────────────────────────
class CreditCardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.net(x)

model = CreditCardModel().to(device)
print(model)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# ── Eğitim ───────────────────────────────────────────────────────────────────
print("\nEğitim başlıyor...\n")
epochs = 20

for epoch in range(1, epochs + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss  = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct    += (preds.argmax(1) == yb.argmax(1)).sum().item()
        total      += xb.size(0)
    print(f"Epoch {epoch:3d}/{epochs} | Loss: {total_loss/total:.4f} | Acc: {correct/total:.4f}")

# ── Değerlendirme ─────────────────────────────────────────────────────────────
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        out = model(xb.to(device))
        all_preds.append(out.argmax(1).cpu().numpy())
        all_true.append(yb.argmax(1).numpy())

test_preds  = np.concatenate(all_preds)
y_test_true = np.concatenate(all_true)

precision = precision_score(y_test_true, test_preds, average="weighted", zero_division=0)
recall    = recall_score   (y_test_true, test_preds, average="weighted", zero_division=0)
f1        = f1_score       (y_test_true, test_preds, average="weighted", zero_division=0)
print(f"\nTest Accuracy : {(test_preds == y_test_true).mean():.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")

torch.save(model.state_dict(), "creditcard_pytorch_model.pt")
print("\nModel 'creditcard_pytorch_model.pt' olarak kaydedildi.")
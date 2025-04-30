import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import pickle

newsgroups = fetch_20newsgroups(subset='all')
X = newsgroups.data
y = newsgroups.target

vectorizer = TfidfVectorizer(stop_words='english', max_features=18500)
X = vectorizer.fit_transform(X).toarray()

with open('tfidf_20news.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

from pyerualjetwork import data_operations, data_operations

x_train, x_test, y_train, y_test = data_operations.split(X, y, test_size=0.2, random_state=42)
y_train, y_test = data_operations.encode_one_hot(y_train, y_test)
x_train, y_train = data_operations.synthetic_augmentation(x_train, y_train)
y_train = data_operations.decode_one_hot(y_train)
y_test = data_operations.decode_one_hot(y_test)

scaler_params, x_train, x_test = data_operations.standard_scaler(x_train, x_test)

device = torch.device("cpu")

x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

batch_size = 64
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

class NewsClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(NewsClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        return x

input_dim = x_train.shape[1]
num_classes = len(newsgroups.target_names)
model = NewsClassifier(input_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)

epochs = 2
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        accuracy_metric.update(outputs, labels)

    epoch_loss = total_loss / len(train_loader)
    epoch_acc = accuracy_metric.compute()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    accuracy_metric.reset()

model.eval()
test_accuracy = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        accuracy_metric.update(outputs, targets)


all_preds = []
all_targets = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.append(preds)
        all_targets.append(targets)

all_preds = torch.cat(all_preds)
all_targets = torch.cat(all_targets)

print("\n------PyTorch Modeli Sonuçları------")
test_accuracy = accuracy_metric.compute()
print(f"Test Accuracy: {test_accuracy:.4f}")
print(classification_report(all_targets.cpu().numpy(), all_preds.cpu().numpy(), target_names=newsgroups.target_names))
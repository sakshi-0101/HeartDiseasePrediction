import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


df = pd.read_csv("heart.csv")

print("Dataset shape:", df.shape)
print(df.head())


df["Sex"] = df["Sex"].map({"M": 0, "F": 1})

df["ChestPainType"] = df["ChestPainType"].map({"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3})

df["RestingECG"] = df["RestingECG"].map({"Normal": 0, "ST": 1, "LVH": 2})

df["ExerciseAngina"] = df["ExerciseAngina"].map({"N": 0, "Y": 1})

df["ST_Slope"] = df["ST_Slope"].map({"Up": 0, "Flat": 1, "Down": 2})


X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]


print("\nFeatures:")
print(X.head())

print("\nTarget distribution:")
print(y.value_counts())


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)

X_test = scaler.transform(X_test)


X_train = torch.tensor(X_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)

y_val = torch.tensor(y_val.to_numpy(), dtype=torch.float32).reshape(-1, 1)

y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).reshape(-1, 1)


print("\nTraining shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Testing shape:", X_test.shape)


# Neural Network Model
class HeartDiseaseNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(11, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4, 1),
        )

    def forward(self, x):
        return self.network(x)


model = HeartDiseaseNN()

print("\nModel:")
print(model)


criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


EPOCHS = 500

# Training Loop
EPOCHS = 1000
PATIENCE = 50

best_val_loss = float("inf")
patience_counter = 0

best_model_state = None

for epoch in tqdm(range(EPOCHS)):

    model.train()

    optimizer.zero_grad()

    train_logits = model(X_train)

    train_loss = criterion(train_logits, y_train)

    train_loss.backward()

    optimizer.step()

    model.eval()

    with torch.no_grad():

        val_logits = model(X_val)

        val_loss = criterion(val_logits, y_val)

    if val_loss.item() < best_val_loss:

        best_val_loss = val_loss.item()

        patience_counter = 0

        best_model_state = {
            key: value.clone() for key, value in model.state_dict().items()
        }

    else:

        patience_counter += 1

    if (epoch + 1) % 25 == 0:

        with torch.no_grad():

            train_probabilities = torch.sigmoid(model(X_train))

            train_predictions = (train_probabilities >= 0.5).float()

            train_accuracy = (train_predictions == y_train).float().mean()

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss.item():.4f} "
            f"Val Loss: {val_loss.item():.4f} "
            f"Train Accuracy: {train_accuracy.item():.4f}"
        )

    if patience_counter >= PATIENCE:

        print(f"\nEarly stopping at epoch {epoch + 1}")

        break


model.load_state_dict(best_model_state)


model.eval()

# Testing
with torch.no_grad():
    test_logits = model(X_test)
    test_probabilities = torch.sigmoid(test_logits)
    test_predictions = (test_probabilities >= 0.5).int()


y_true = y_test.numpy().flatten()
y_pred = test_predictions.numpy().flatten()
y_prob = test_probabilities.numpy().flatten()


accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

roc_auc = roc_auc_score(y_true, y_prob)


print("\n")
print("*" * 30)
print("Neural Network Results")
print("*" * 30)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

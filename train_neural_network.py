import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train = torch.tensor(X_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)

y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).reshape(-1, 1)


print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

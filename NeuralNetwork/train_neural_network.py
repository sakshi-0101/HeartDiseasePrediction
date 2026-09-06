import random
import numpy as np

import torch
import torch.nn as nn


from neural_network import HeartDiseaseNN
from train import train
from test import test
from score import score
from save_model import save_model
from data_preprocessing import preporcessing
from totensor import totensor

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

X_train, X_val, X_test, y_train, y_val, y_test, scaler = preporcessing(SEED)
X_train, X_val, X_test, y_train, y_val, y_test = totensor(
    X_train, X_val, X_test, y_train, y_val, y_test
)

print("\nTraining shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Testing shape:", X_test.shape)


model = HeartDiseaseNN()

print("\nModel:")
print(model)


criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
best_model_state = train(model, criterion, optimizer, X_train, X_val, y_train, y_val)

model.load_state_dict(best_model_state)


test_predictions, test_probabilities = test(model, X_test)

score(y_test, test_predictions, test_probabilities)

save_model(model, scaler)

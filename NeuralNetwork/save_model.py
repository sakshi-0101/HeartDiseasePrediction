import os
import joblib
import torch


def save_model(model, scaler):
    os.makedirs("../models", exist_ok=True)

    torch.save(model.state_dict(), "../models/heart_disease_nn.pth")

    joblib.dump(scaler, "../models/neural_network_scaler.pkl")

    print("\nModel saved successfully.")
    print("Scaler saved successfully.")

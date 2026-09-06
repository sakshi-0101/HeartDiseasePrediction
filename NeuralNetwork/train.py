import torch
from torch import nn
from tqdm.auto import tqdm


def train(model, criterion, optimizer, X_train, X_val, y_train, y_val):
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

    return best_model_state

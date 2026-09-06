import torch


def totensor(X_train, X_val, X_test, y_train, y_val, y_test):
    X_train = torch.tensor(X_train, dtype=torch.float32)

    X_val = torch.tensor(X_val, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)

    y_val = torch.tensor(y_val.to_numpy(), dtype=torch.float32).reshape(-1, 1)

    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).reshape(-1, 1)

    return X_train, X_val, X_test, y_train, y_val, y_test

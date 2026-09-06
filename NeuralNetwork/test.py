import torch


def test(model, X_test):
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        test_probabilities = torch.sigmoid(test_logits)
        test_predictions = (test_probabilities >= 0.5).int()
    return test_predictions, test_probabilities

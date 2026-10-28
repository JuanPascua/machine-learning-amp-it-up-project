import numpy as np

#Calculates the means squared error for the actual target values for a target values and the predicted values for a subset
def mse(y, y_hat):
    diff = np.asarray(y, dtype=np.float64) - np.asarray(y_hat, dtype=np.float64)
    return float(np.mean(diff ** 2))

#Calculates the error to signal ratio by taking the summation of the diffrence between the predicted and target y squared over the summation of the target y squared for a subset
def esr(y, y_hat):
    """Error-to-signal ratio: squared error divided by the energy of the target."""
    y = np.asarray(y, dtype=np.float64)
    diff = y - np.asarray(y_hat, dtype=np.float64)
    return float(np.sum(diff ** 2) / np.sum(y ** 2))

# Create a report/dictionary of the mse and esr
def report(y, y_hat):
    return {"mse": mse(y, y_hat), "esr": esr(y, y_hat)}

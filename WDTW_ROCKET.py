# ==========================================================
# WDTW vs ROCKET
# Time Series Forecasting on BasicMotions
# ==========================================================

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV

from sktime.datasets import load_UCR_UEA_dataset
from sktime.transformations.panel.rocket import MiniRocketMultivariate


# ==========================================================
# Reproducibility
# ==========================================================

np.random.seed(42)
torch.manual_seed(42)


# ==========================================================
# Load UCR/UEA Dataset
# ==========================================================

X_train_raw, y_train_raw = load_UCR_UEA_dataset(
    name="BasicMotions",
    split="train",
    return_type="numpy3d"
)

X_test_raw, y_test_raw = load_UCR_UEA_dataset(
    name="BasicMotions",
    split="test",
    return_type="numpy3d"
)

print("Train shape:", X_train_raw.shape)
print("Test shape:", X_test_raw.shape)

# Shape:
# (n_samples, n_channels, series_length)


# ==========================================================
# Convert classification dataset -> forecasting stream
# ==========================================================

# Use first channel only
train_series_collection = X_train_raw[:, 0, :]
test_series_collection = X_test_raw[:, 0, :]

train_stream = []
test_stream = []

for s in train_series_collection:
    train_stream.extend(s)

for s in test_series_collection:
    test_stream.extend(s)

train_stream = np.array(train_stream)
test_stream = np.array(test_stream)

print("Train stream shape:", train_stream.shape)
print("Test stream shape:", test_stream.shape)


# ==========================================================
# Sliding windows
# ==========================================================

WINDOW = 40
HORIZON = 1


def create_dataset(series, window, horizon=1):

    X = []
    y = []

    for i in range(len(series) - window - horizon):

        X.append(series[i:i + window])

        y.append(series[i + window:i + window + horizon])

    return np.array(X), np.array(y)


X_train, y_train = create_dataset(
    train_stream,
    WINDOW,
    HORIZON
)

X_test, y_test = create_dataset(
    test_stream,
    WINDOW,
    HORIZON
)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ==========================================================
# Weighted Dynamic Time Warping
# ==========================================================

class WDTW:

    def __init__(self, g=0.05):
        self.g = g

    def weight(self, diff, m):

        return 1.0 / (
            1.0 + np.exp(-self.g * (diff - m))
        )

    def distance(self, ts1, ts2):

        n = len(ts1)
        m = len(ts2)

        cost = np.full((n + 1, m + 1), np.inf)

        cost[0, 0] = 0

        midpoint = max(n, m) / 2

        for i in range(1, n + 1):

            for j in range(1, m + 1):

                weight = self.weight(
                    abs(i - j),
                    midpoint
                )

                dist = weight * (
                    (ts1[i - 1] - ts2[j - 1]) ** 2
                )

                cost[i, j] = dist + min(
                    cost[i - 1, j],
                    cost[i, j - 1],
                    cost[i - 1, j - 1]
                )

        return np.sqrt(cost[n, m])


# ==========================================================
# WDTW Forecasting Model
# ==========================================================

class WDTWForecaster:

    def __init__(self, k=3, g=0.05):

        self.k = k
        self.wdtw = WDTW(g=g)

    def fit(self, X, y):

        self.X = X
        self.y = y

    def predict_one(self, query):

        distances = []

        for i in range(len(self.X)):

            d = self.wdtw.distance(
                query,
                self.X[i]
            )

            distances.append((d, i))

        distances.sort(key=lambda x: x[0])

        neighbors = distances[:self.k]

        preds = []

        for _, idx in neighbors:
            preds.append(self.y[idx])

        return np.mean(preds, axis=0)

    def predict(self, X):

        predictions = []

        for sample in X:

            pred = self.predict_one(sample)

            predictions.append(pred)

        return np.array(predictions)


# ==========================================================
# Train WDTW
# ==========================================================

wdtw_model = WDTWForecaster(
    k=3,
    g=0.1
)

wdtw_model.fit(
    X_train,
    y_train
)

wdtw_preds = wdtw_model.predict(X_test)

wdtw_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        wdtw_preds
    )
)

print("\nWDTW RMSE:", wdtw_rmse)


# ==========================================================
# ROCKET Baseline
# ==========================================================

# ROCKET expects:
# (n_instances, n_channels, series_length)

X_train_rocket = X_train[:, np.newaxis, :]
X_test_rocket = X_test[:, np.newaxis, :]

print("\nROCKET input shape:", X_train_rocket.shape)


# ==========================================================
# MiniROCKET Feature Extraction
# ==========================================================

rocket = MiniRocketMultivariate(
    random_state=42
)

rocket.fit(X_train_rocket)

X_train_transform = rocket.transform(
    X_train_rocket
)

X_test_transform = rocket.transform(
    X_test_rocket
)

print(
    "ROCKET features:",
    X_train_transform.shape
)


# ==========================================================
# Ridge Regression
# ==========================================================

ridge = RidgeCV(
    alphas=np.logspace(-3, 3, 10)
)

ridge.fit(
    X_train_transform,
    y_train.ravel()
)

rocket_preds = ridge.predict(
    X_test_transform
)

rocket_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rocket_preds
    )
)

print("ROCKET RMSE:", rocket_rmse)


# ==========================================================
# Torch WDTW Module
# ==========================================================

class TorchWDTW(nn.Module):

    def __init__(self, g=0.05):

        super().__init__()

        self.g = g

    def weight(self, diff, m):

        return 1.0 / (
            1.0 + torch.exp(
                -self.g * (diff - m)
            )
        )

    def forward(self, x, y):

        n = x.shape[0]
        m = y.shape[0]

        cost = torch.full(
            (n + 1, m + 1),
            float("inf")
        )

        cost[0, 0] = 0

        midpoint = max(n, m) / 2

        for i in range(1, n + 1):

            for j in range(1, m + 1):

                diff = abs(i - j)

                weight = self.weight(
                    torch.tensor(diff).float(),
                    midpoint
                )

                dist = weight * (
                    x[i - 1] - y[j - 1]
                ) ** 2

                cost[i, j] = dist + torch.min(
                    torch.stack([
                        cost[i - 1, j],
                        cost[i, j - 1],
                        cost[i - 1, j - 1]
                    ])
                )

        return torch.sqrt(cost[n, m])


# ==========================================================
# Example Torch WDTW Usage
# ==========================================================

torch_wdtw = TorchWDTW(g=0.1)

x1 = torch.tensor(X_train[0]).float()
x2 = torch.tensor(X_train[1]).float()

distance = torch_wdtw(x1, x2)

print("\nTorch WDTW distance:", distance.item())


# ==========================================================
# Visualization
# ==========================================================

plt.figure(figsize=(14, 6))

plt.plot(
    y_test[:200],
    label="True"
)

plt.plot(
    wdtw_preds[:200],
    label="WDTW"
)

plt.plot(
    rocket_preds[:200],
    label="ROCKET"
)

plt.title(
    "Forecasting Comparison: WDTW vs ROCKET"
)

plt.legend()

plt.show()


# ==========================================================
# Final Summary
# ==========================================================

print("\n==============================")
print("FINAL RESULTS")
print("==============================")
print(f"WDTW RMSE   : {wdtw_rmse:.6f}")
print(f"ROCKET RMSE : {rocket_rmse:.6f}")
print("==============================")
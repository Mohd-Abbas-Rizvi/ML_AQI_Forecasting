# From-scratch implementations of Ridge Regression and Multi-Layer Perceptron in NumPy
import numpy as np
from typing import Dict, Optional, Tuple


class NumPyRidgeRegression:
    # Closed-form Ridge Regression implemented from scratch in NumPy
    def __init__(self, alpha: float = 1.0):
        self.name = "NumPy_Ridge"
        self.alpha = float(alpha)
        self.weights = None
        self.bias = 0.0
        self.mean_ = None
        self.scale_ = None
        self.col_medians_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        # Handle NaNs in target y
        valid_y = ~np.isnan(y)
        X = X[valid_y]
        y = y[valid_y]

        # Impute missing feature values with train median
        self.col_medians_ = np.nanmedian(X, axis=0)
        self.col_medians_[np.isnan(self.col_medians_)] = 0.0
        
        inds = np.where(np.isnan(X))
        X[inds] = np.take(self.col_medians_, inds[1])

        # Standardize features
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0

        X_norm = (X - self.mean_) / self.scale_
        y_mean = np.mean(y)
        y_centered = y - y_mean

        n_samples, n_features = X_norm.shape
        # Ridge normal equation: (X^T X + alpha * I)^{-1} X^T y
        A = X_norm.T @ X_norm + self.alpha * np.eye(n_features)
        b = X_norm.T @ y_centered

        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(A) @ b

        self.weights = w
        self.bias = float(y_mean)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float).copy()
        inds = np.where(np.isnan(X))
        X[inds] = np.take(self.col_medians_, inds[1])
        X_norm = (X - self.mean_) / self.scale_
        return (X_norm @ self.weights) + self.bias


class NumPyMLPRegressor:
    # 2-layer Neural Network MLP implemented from scratch in NumPy with Adam optimizer
    def __init__(
        self,
        hidden_dim: int = 64,
        lr: float = 0.005,
        alpha: float = 0.001,
        epochs: int = 150,
        batch_size: int = 64,
        random_state: int = 42,
    ):
        self.name = "NumPy_MLP"
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.mean_ = None
        self.scale_ = None
        self.col_medians_ = None
        self.y_mean = 0.0
        self.y_std = 1.0

    def _relu(self, z):
        return np.maximum(0.0, z)

    def _relu_grad(self, z):
        return (z > 0.0).astype(float)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        np.random.seed(self.random_state)
        X = np.asarray(X, dtype=float).copy()
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        valid_y = ~np.isnan(y).ravel()
        X = X[valid_y]
        y = y[valid_y]

        self.col_medians_ = np.nanmedian(X, axis=0)
        self.col_medians_[np.isnan(self.col_medians_)] = 0.0
        inds = np.where(np.isnan(X))
        X[inds] = np.take(self.col_medians_, inds[1])

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0

        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y))
        if self.y_std == 0.0:
            self.y_std = 1.0

        X_norm = (X - self.mean_) / self.scale_
        y_norm = (y - self.y_mean) / self.y_std

        n_samples, n_features = X_norm.shape

        # He initialization
        self.W1 = np.random.randn(n_features, self.hidden_dim) * np.sqrt(2.0 / n_features)
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = np.random.randn(self.hidden_dim, 1) * np.sqrt(2.0 / self.hidden_dim)
        self.b2 = np.zeros((1, 1))

        # Adam state
        mW1, vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        mb1, vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        mW2, vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        mb2, vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)

        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_norm[indices]
            y_shuffled = y_norm[indices]

            for i in range(0, n_samples, self.batch_size):
                xb = X_shuffled[i : i + self.batch_size]
                yb = y_shuffled[i : i + self.batch_size]
                m = len(xb)
                if m == 0:
                    continue

                t += 1
                # Forward pass
                z1 = xb @ self.W1 + self.b1
                a1 = self._relu(z1)
                y_pred_b = a1 @ self.W2 + self.b2

                # Backward pass
                err = y_pred_b - yb
                dW2 = (a1.T @ err) / m + self.alpha * self.W2
                db2 = np.sum(err, axis=0, keepdims=True) / m

                da1 = err @ self.W2.T
                dz1 = da1 * self._relu_grad(z1)
                dW1 = (xb.T @ dz1) / m + self.alpha * self.W1
                db1 = np.sum(dz1, axis=0, keepdims=True) / m

                # Adam updates
                for param, dparam, m_arr, v_arr in [
                    (self.W1, dW1, mW1, vW1),
                    (self.b1, db1, mb1, vb1),
                    (self.W2, dW2, mW2, vW2),
                    (self.b2, db2, mb2, vb2),
                ]:
                    m_arr[:] = beta1 * m_arr + (1 - beta1) * dparam
                    v_arr[:] = beta2 * v_arr + (1 - beta2) * (dparam ** 2)
                    m_hat = m_arr / (1.0 - beta1 ** t)
                    v_hat = v_arr / (1.0 - beta2 ** t)
                    param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float).copy()
        inds = np.where(np.isnan(X))
        X[inds] = np.take(self.col_medians_, inds[1])
        X_norm = (X - self.mean_) / self.scale_
        z1 = X_norm @ self.W1 + self.b1
        a1 = self._relu(z1)
        out_norm = a1 @ self.W2 + self.b2
        out = (out_norm * self.y_std) + self.y_mean
        return out.ravel()

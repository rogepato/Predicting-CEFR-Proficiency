import numpy as np
from typing import Tuple, Optional

class SoftmaxRegression:
    def __init__(
        self,
        n_features: int = 10,
        n_classes: int = 5,
        learning_rate: float = 0.1,
        num_epochs: int = 100,
        reg_lambda: float = 0.0
    ):

        self.n_features = n_features
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.reg_lambda = reg_lambda

        # Model parameters
        self.W: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None

        # track loss over epochs
        self.loss_history = []

    def _softmax(self, Z: np.ndarray) -> np.ndarray:
        """
        Z: matrix representing logits, each row corresponds to 
           an example, each column represents one class

        returns: softmax probabilities
        """
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)
        P = exp_Z / sum_exp_Z
        return P
        

    def _one_hot(self, y: np.ndarray) -> np.ndarray:
        """
        y: label vector, each value represents which class it is,
           using 0-index counting

        returns: each row has a 1 in the column that corresponds
                 the class labels, 0 elsewhere
        """
        N = y.shape[0]
        Y = np.zeros((N, self.n_classes), dtype=np.float32)

        for i in range(N):
            class_index = int(y[i])
            Y[i, class_index] = 1

        return Y

    def _compute_loss(self, X:np.ndarray, y: np.ndarray) -> float:
        """
        Compute cross-entropy loss.
        Use Lasso regularization if reg_lambda > 0.0.

        params:
                X: input data, N x n_features
                y: class labels, N x ,

        returns:
                float cross_entropy loss
        """
        N = X.shape[0]

        Z = X @ self.W + self.b
        P = self._softmax(Z)
        Y = self._one_hot(y)
        eps = 1e-15
        log_P = np.log(P + eps)
        data_loss = -np.sum(Y * log_P) / N

        # l2 regularization on W
        if self.reg_lambda > 0.0:
            reg_loss = (self.reg_lambda / 2.0) * np.sum(self.W * self.W)
        else:
            reg_loss = 0.0

        return float(data_loss + reg_loss)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        train softmax regression model using batch gradient descent.

        params:
                X: input data, N x n_features
                y: class labels, N x ,  
        """
        N, d = X.shape

        if self.W is None:
            self.W = 0.01 * np.random.randn(self.n_features, self.n_classes)
        if self.b is None:
            self.b = np.zeros(self.n_classes, dtype=np.float64)

        for epoch in range(self.num_epochs):
            Z = X @ self.W + self.b
            P = self._softmax(Z)
            Y = self._one_hot(y)
        
            # compute gradients:
            #    dW = (1/N) * X^T (P - Y) + reg_lambda * W
            #    db = (1/N) * sum_rows(P - Y)
            E = P - Y
            dW = (X.T @ E) / N
            db = np.sum(E, axis=0) / N
    
            # L2 regularization on dW
            if self.reg_lambda > 0.0:
                dW += self.reg_lambda * self.W
    
            # update param
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
    
            # keep track of loss history
            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        predict class probabilities for each instance in X
        params:
            X: input data, N x n_features
            y: class labels, N x 
        returns:
            predicted probabilites of shape (N, n_classes)
        """
        Z = X @ self.W + self.b
        P = self._softmax(Z)
        return P

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        predict class labels as integers for each instance in X
        
        params:
            X: input data, N x n_features

        returns:
            predicted labels of shape (N, ) with class values using 0-indexed
        """
        P = self.predict_proba(X)
        y_pred = np.argmax(P, axis=1)
        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        compute classification accuracy
        
        params:
                X: input data, N x n_features
                y: class labels, N x ,  

        returns:
            accuracy as a float
        """
        y_pred = self.predict(X)
        correct = np.sum(y_pred == y)
        N = X.shape[0]
        return float(correct / N)
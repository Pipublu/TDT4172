import numpy as np

class LinearRegression():
    
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.learningRate = learning_rate;
        self.w = None
        self.b = None
        self.n_iters = n_iters
        
    def fit(self, X, y):
        # Initialize
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        m,n = X.shape
        self.w = np.zeros(n) 
        self.b = 0  
        
        # Iterate
        for _ in range(self.n_iters):
            y_pred = self.predict(X) # Predict y values
            
            # Calculate gradients
            dw = (2/m) * X.T.dot(y_pred - y)
            db = (2/m) * np.sum(y_pred - y)
            
            # Update parameters
            self.w -= self.learningRate * dw
            self.b -= self.learningRate * db
            
        return
        
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X.dot(self.w) + self.b



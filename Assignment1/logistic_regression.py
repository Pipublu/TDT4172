import numpy as np

class LogisticRegression():
  
  def __init__(self, learning_rate=0.0005, epochs=500):
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.w, self.b = None, None
    self.loss, self.train_accuracies = [], []
    self.pred_to_class = []
    
  def fit(self, x, y):
    self.w = np.zeros(x.shape[1])
    self.b = 0
    for _ in range(self.epochs):
      lin_model = np.matmul(self.w, x.transpose()) + self.b
      y_pred = self._sigmoid(lin_model) # Calculate probabilty
      grad_w, grad_b = self.compute_gradients(x, y, y_pred)
      self.update_parameters(grad_w, grad_b)
      self.loss.append(self._compute_loss(y, y_pred))
      self.pred_to_class = [1 if _y > 0.5 else 0 for _y in y_pred] # Classify prediction
      self.train_accuracies.append(self.accuracy(y, self.pred_to_class))
      
  def accuracy(self, true_values, predictions):
    return np.mean(true_values == predictions)
  
  def predict(self, x):
    lin_model = np.matmul(x, self.w) + self.b
    y_pred = self._sigmoid(lin_model)
    return [1 if _y > 0.5 else 0 for _y in y_pred]
  
  def _sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
  
  def _compute_loss(self, y, y_pred): # Binary cross-entropy loss
    n_samples = len(y)
    loss = - (1/n_samples) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return loss
  
  def compute_gradients(self, x, y, y_pred):
    n_samples = x.shape[0]
    grad_w = (1/n_samples) * np.dot((y_pred - y), x)
    grad_b = (1/n_samples) * np.sum(y_pred - y)
    return grad_w, grad_b
  
  def update_parameters(self, grad_w, grad_b):
    self.w -= self.learning_rate * grad_w
    self.b -= self.learning_rate * grad_b
    
  def predict_proba(self, x):
    lin_model = np.matmul(x, self.w) + self.b
    y_pred = self._sigmoid(lin_model)
    return y_pred


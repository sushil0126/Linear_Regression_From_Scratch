import numpy as np

# Add a column of ones to X to represent the bias (intercept) term
def add_Bias(X):
    ones = np.ones((X.shape[0], 1))
    X_hat = np.hstack((ones, X))
    return X_hat

# Train the model using gradient descent to learn the optimal theta
def training_model(X, y, theta):
    m = X.shape[0]
    learning_rate = 0.0001
    iterations = 10000

    X_hat = add_Bias(X)

    for i in range(iterations):
        y_hat = X_hat @ theta
        residual = y_hat - y
        mse = (1 / m) * np.sum(residual ** 2)
        gradient = (2 / m) * (X_hat.T @ residual)
        theta = theta - learning_rate * gradient

        if i % 1000 == 0:
            print(f"Iteration {i}, MSE = {mse}")

    return theta 

# Generate predictions using learned parameters
def predict_values(X, theta):
    X_hat = add_Bias(X)
    return X_hat @ theta

# Train the model and return predictions and learned theta
def train_Predict(X, y, theta):
    theta = training_model(X, y, theta)
    y_pred = predict_values(X, theta)
    return y_pred, theta 

# Compute R² score to evaluate model performance
def r2_score(y_actual, y_pred):
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)

    r2 = 1 - (ss_res / ss_tot)
    return r2

X = np.array([
[10,13,9],
[12,15,11],
[14,18,13],
[16,20,15],
[18,23,17],
[20,25,19],
[22,28,21],
[24,30,23],
[26,33,25],
[28,35,27],
[30,38,29],
[32,40,31],
[34,43,33],
[36,45,35],
[38,48,37],
])

y = np.array([
[12],
[14],
[16],
[18],
[20],
[22],
[24],
[26],
[28],
[30],
[32],
[34],
[36],
[38],
[40],
])

# Initialize theta (weights) including bias
theta = np.zeros((X.shape[1] + 1, 1))

# Train model and get predictions
Y_pred, trained_theta = train_Predict(X, y, theta)

print("\nLearned theta:")
print(trained_theta) 

print("\nPredictions:")
print(Y_pred)  

# Evaluate model using R² score
r2 = r2_score(y, Y_pred)
print("\nR2 Score:", r2)


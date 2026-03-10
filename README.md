# Linear_Regression  

# Understanding Machine Learning by Implementing Linear Regression from Scratch with NumPy

## Project Description
This project demonstrates how to implement a **Linear Regression model from scratch using only NumPy** without using machine learning libraries such as scikit-learn. The purpose of this project is to understand the **core mathematical and computational principles behind machine learning algorithms**.

The model learns relationships between input features and the target variable using **Gradient Descent optimization**. The performance of the model is evaluated using the **R² score**, which measures how well the model explains the variance in the data.

---

## Objectives

The main objectives of this project are:

- Understand how **Linear Regression works internally**
- Implement **Gradient Descent optimization**
- Compute **Mean Squared Error (MSE)** during training
- Evaluate model performance using **R² Score**
- Practice **NumPy matrix operations**

---

## Technologies Used

- Python  
- NumPy  

---

## Project Structure

```
linear-regression-from-scratch/
│
├── linear_regression.py
└── README.md
```

---

## How the Model Works

The implementation follows these steps:

1. Add a **bias (intercept) term** to the feature matrix.
2. Initialize the model parameters (**theta**).
3. Train the model using **Gradient Descent**.
4. Calculate the **Mean Squared Error** during training.
5. Generate predictions using the trained parameters.
6. Evaluate the model using the **R² Score**.

---

## Mathematical Formulation

### Linear Regression Equation

y = Xθ

Where:

- X = Feature matrix  
- θ = Model parameters  
- y = Predicted output  

---

### Mean Squared Error (MSE)

MSE = (1/m) Σ (y_pred − y)²

---

### Gradient Descent Update Rule

θ = θ − α ∇J(θ)

Where:

- α = learning rate  
- J(θ) = cost function  

---

## Example Output

```
Iteration 0, MSE = 750.6666666666666
Iteration 1000, MSE = 0.5096093017834276
Iteration 2000, MSE = 0.4685078238207818
Iteration 3000, MSE = 0.43103894544312094


Learned theta:
[[0.4196619],
 [0.39447638],
 [0.54611144],
 [-0.02518553]]

Predictions:
[[11.23720465],
 [13.06800923],
 [15.44492525],
 [17.27572983]]

R2 Score: 0.99669802279

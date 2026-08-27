# Lecture 1: Linear Regression (sample)

This is a short synthetic note used to demo the TAlker pipeline. It is not
course material from any real class.

Linear regression models a target y as a weighted sum of input features:
y = w0 + w1*x1 + ... + wp*xp. The weights are chosen to minimise the sum of
squared residuals between predictions and observed values.

Key points:

- Ordinary least squares has a closed-form solution when X^T X is invertible.
- Gradient descent is used instead when the feature count is large.
- R^2 measures the share of variance in y explained by the model.
- Add polynomial features to fit curves while keeping the model linear in
  its parameters.

Assignment 1 (regression on the housing dataset) is due Friday at 23:59.

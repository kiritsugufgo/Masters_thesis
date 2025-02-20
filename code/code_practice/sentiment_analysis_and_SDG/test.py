import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from mpi4py import MPI

# Load dataset
data = pd.read_csv("data.csv")

# Split data into train and test
data = data.dropna(subset=['IMDB_Rating', 'Overview', 'Released_Year', 'Runtime', 'Genre', 'Meta_score', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes', 'Gross', 'Certificate'])

# Extract numerical values from the 'Runtime' column
data['Runtime'] = data['Runtime'].str.extract('(\d+)').astype(int)

# Extract features and target
X_text = data['Overview']
y = data['IMDB_Rating']

# Preprocess text data
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X_text_vec = vectorizer.fit_transform(X_text).toarray()

# Preprocess additional features
additional_features = data[['Released_Year', 'Runtime', 'Genre', 'Meta_score', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes', 'Gross']]

# Convert categorical features to numerical using one-hot encoding
categorical_features = ['Genre', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'Certificate']
one_hot_encoder = OneHotEncoder()
X_categorical = one_hot_encoder.fit_transform(additional_features[categorical_features]).toarray()

# Normalize numerical features
numerical_features = ['Released_Year', 'Runtime', 'Meta_score', 'No_of_Votes', 'Gross']
scaler = StandardScaler()
X_numerical = scaler.fit_transform(additional_features[numerical_features])

# Combine all features
X_combined = np.hstack((X_text_vec, X_categorical, X_numerical))

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Binary cross-entropy loss
def loss_function(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-10) + (1 - y_true) * np.log(1 - y_pred + 1e-10))

# Gradient of the loss
def compute_gradients(X, y_true, weights):
    m = X.shape[0]
    y_pred = sigmoid(np.dot(X, weights))
    error = y_pred - y_true
    gradients = np.dot(X.T, error) / m
    return gradients

# SGD training
def train_sgd(X, y, num_features, learning_rate=0.01, epochs=10):
    weights = np.zeros(num_features)  # Initialize weights
    for epoch in range(epochs):
        gradients = compute_gradients(X, y, weights)
        weights -= learning_rate * gradients  # Update weights
        y_pred = sigmoid(np.dot(X, weights))
        loss = loss_function(y, y_pred)
        print(f"Epoch {epoch + 1}, Loss: {loss}")
    return weights

# Parameters
num_features = X_train_vec.shape[1]
learning_rate = 0.01
epochs = 10

# Train the model
weights = train_sgd(X_train_vec, y_train_enc, num_features, learning_rate, epochs)

# Evaluate the model
def evaluate(X, y, weights):
    y_pred = sigmoid(np.dot(X, weights))
    y_pred_class = (y_pred > 0.5).astype(int)
    accuracy = np.mean(y_pred_class == y)
    return accuracy

accuracy = evaluate(X_test_vec, y_test_enc, weights)
print(f"Test Accuracy: {accuracy}")
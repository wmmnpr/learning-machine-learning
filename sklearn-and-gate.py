import numpy as np
from sklearn.neural_network import MLPClassifier

# Define the AND gate truth table
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input features
y = np.array([0, 0, 0, 1])  # Output labels

# Create a single-layer perceptron (MLP) neural network
# The activation function used here is 'identity' which is equivalent to a linear activation.
# The solver is set to 'lbfgs', but you can experiment with other solvers.
model = MLPClassifier(hidden_layer_sizes=(1,), activation='identity', solver='lbfgs', random_state=42)

# Train the neural network
model.fit(X, y)

# Test the trained model
test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = model.predict(test_input)

# Display the results
for i in range(len(test_input)):
    print(f"Input: {test_input[i]}, Predicted Output: {predictions[i]}")
    
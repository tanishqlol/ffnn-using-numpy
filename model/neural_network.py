import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float) 

def sigmoid(x):
    # clip x to prevent overflow and for efficient computation
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

# we will define our feed forward neural network using a Python class
class Network:
    def __init__(self, layers: list[int]):
        np.random.seed(1)
        
        #initializing weights and biases
        self.layers = layers # layers is a list (e.g., [2, 5, 4] cooresponds to 2 input, 5 neurons in hidden layer, 4 neurons in output layer)
        self.biases = [np.random.randn(i, 1) for i in layers[1:]]
        self.weights = [np.random.randn(k, j) for j, k in zip(layers[:-1], layers[1:])]
        
        # store activation and z values to be used during backpropagation
        self.activations = []
        self.z_values = []

    def forward_pass(self, X):
        """
        forward pass through the network
        X shape: (input_features, batch_size)
        Returns: (output_features, batch_size)
        """
        self.activations = [X]  # store input as first activation
        self.z_values = []
        
        A = X
        iterations = len(self.layers) - 1 # no. of iterations until the penultimate layer
        
        # hidden layers with ReLU
        for i in range(iterations - 1):
            Z = self.weights[i] @ A + self.biases[i]
            self.z_values.append(Z)
            A = relu(Z)
            self.activations.append(A)
        
        # output layer with sigmoid
        Z = self.weights[-1] @ A + self.biases[-1]
        self.z_values.append(Z)
        A = sigmoid(Z)
        self.activations.append(A)
        
        return A
    
    def one_hot_encode(self, labels):
        """
        convert labels to one-hot encoding
        labels: 1D array of shape (batch_size,)
        returns: 2D array of shape (num_classes, batch_size)
        """
        num_classes = self.layers[-1]  # output layer size
        batch_size = labels.shape[0]
        
        y = np.zeros((num_classes, batch_size))
        y[labels, np.arange(batch_size)] = 1 # assigning 1 to the exact location using advanced numpy indexing
        return y

    def categorical_cross_entropy_loss(self, y_true, y_pred):
        """
        calculate categorical cross-entropy loss
        y_true, y_pred: shape (num_classes, batch_size)
        """

        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Calculate loss
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[1]
        return loss

    def backward_pass(self, X, y_true):
        """
        backpropagation to compute gradients
        X: input data of shape (input_features, batch_size)
        y_true: true labels, shape (num_classes, batch_size) - one-hot encoded
        """
        batch_size = X.shape[1]
        num_layers = len(self.layers)
        
        # initialize gradient storage with 0s (to be updated with when gradients are calculated)
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # start with output layer

        # for sigmoid with cross-entropy, chain rule for derivatives gives a simple formula: y_pred - y_true
        y_pred = self.activations[-1]
        dZ = y_pred - y_true
        
        # Backpropagate through each layer
        for layer in reversed(range(num_layers - 1)):
            # Current layer gradients
            dW[layer] = (1/batch_size) * dZ @ self.activations[layer].T
            db[layer] = (1/batch_size) * np.sum(dZ, axis=1, keepdims=True)
            
            # Don't compute dA for input layer
            if layer > 0:
                # Gradient w.r.t. previous layer's activation
                dA_prev = self.weights[layer].T @ dZ
                
                # Apply derivative of activation function
                if layer > 0:  # Hidden layers use ReLU
                    dZ = dA_prev * relu_derivative(self.z_values[layer-1])
        
        return dW, db

    def update_batch_parameters(self, dW, db, learning_rate):
        """
        update weights and biases using computed gradients
        """
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dW[i]
            self.biases[i] -= learning_rate * db[i]

    def SGD(self, X, y, learning_rate, batch_size=32):
        """
        Stochastic Gradient Descent for one epoch
        X: input data, shape (input_features, num_samples)
        y: labels, shape (num_samples,)
        """
        num_samples = X.shape[1]
        
        # Shuffle data
        indices = np.random.permutation(num_samples)
        X_shuffled = X[:, indices]
        y_shuffled = y[indices]
        
        total_loss = 0
        num_batches = 0
        
        # Process in mini-batches
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            X_batch = X_shuffled[:, i:end_idx]
            y_batch = y_shuffled[i:end_idx]
            
            # Convert labels to one-hot
            y_one_hot = self.one_hot_encode(y_batch)
            
            # Forward pass
            y_pred = self.forward_pass(X_batch)
            
            # Calculate loss
            batch_loss = self.categorical_cross_entropy_loss(y_one_hot, y_pred)
            total_loss += batch_loss
            num_batches += 1
            
            # Backward pass
            dW, db = self.backward_pass(X_batch, y_one_hot)
            
            # Update parameters
            self.update_batch_parameters(dW, db, learning_rate)
        
        return total_loss / num_batches

    def fit(self, X, y, epochs, learning_rate=0.01, batch_size=32, verbose=True):
        """
        train the neural network
        X: input data, shape (input_features, num_samples)
        y: labels, shape (num_samples,)
        """
        for epoch in range(epochs):
            avg_loss = self.SGD(X, y, learning_rate, batch_size)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def predict(self, X):
        """
        Make predictions
        X: input data, shape (input_features, num_samples)
        Returns: predicted class labels, shape (num_samples,)
        """
        y_pred = self.forward_pass(X)
        return np.argmax(y_pred, axis=0)

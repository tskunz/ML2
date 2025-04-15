# Practical Neural Network Implementation with Keras

## Introduction
This guide provides a complete, practical example of implementing a neural network using Keras. We'll cover data preparation, model building, training, evaluation, and visualization.

## Complete Implementation

### 1. Setup and Dependencies
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

### 2. Data Preparation
```python
def prepare_data(X, y, test_size=0.2):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Example usage with MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape((X_train.shape[0], -1)) / 255.0
X_test = X_test.reshape((X_test.shape[0], -1)) / 255.0
```

### 3. Model Architecture
```python
def create_model(input_shape, num_classes):
    model = keras.Sequential([
        # Input layer
        keras.layers.Dense(
            256, 
            activation='relu',
            kernel_initializer='he_normal',
            input_shape=input_shape
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        # Hidden layers
        keras.layers.Dense(
            128, 
            activation='relu',
            kernel_initializer='he_normal'
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(
            64, 
            activation='relu',
            kernel_initializer='he_normal'
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        # Output layer
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
```

### 4. Training Configuration
```python
def configure_training(model, learning_rate=1e-3):
    # Learning rate schedule
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=0.9
    )
    
    # Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True
        ),
        keras.callbacks.TensorBoard(log_dir='./logs')
    ]
    
    return callbacks
```

### 5. Training and Evaluation
```python
def train_and_evaluate(model, X_train, y_train, X_test, y_test, callbacks):
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    return history

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
```

### 6. Model Analysis
```python
def analyze_model_predictions(model, X_test, y_test):
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    # Classification report
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))
```

### 7. Complete Example Usage
```python
def main():
    # Load and prepare data
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], -1)) / 255.0
    X_test = X_test.reshape((X_test.shape[0], -1)) / 255.0
    
    # Create model
    input_shape = (784,)  # 28x28 pixels flattened
    num_classes = 10
    model = create_model(input_shape, num_classes)
    
    # Configure training
    callbacks = configure_training(model)
    
    # Train and evaluate
    history = train_and_evaluate(
        model, X_train, y_train, X_test, y_test, callbacks
    )
    
    # Plot results
    plot_training_history(history)
    
    # Analyze predictions
    analyze_model_predictions(model, X_test, y_test)

if __name__ == "__main__":
    main()
```

## Best Practices Implemented

1. **Data Preprocessing**
   - Scaling/normalization
   - Train/test split
   - Data validation

2. **Model Architecture**
   - Proper initialization
   - Batch normalization
   - Dropout for regularization
   - Appropriate activation functions

3. **Training Configuration**
   - Learning rate scheduling
   - Early stopping
   - Model checkpointing
   - TensorBoard logging

4. **Evaluation**
   - Multiple metrics
   - Confusion matrix
   - Classification report
   - Visual analysis

## Common Issues and Solutions

### 1. Overfitting
```python
# Add more regularization
model.add(keras.layers.Dense(
    64,
    kernel_regularizer=keras.regularizers.l2(0.01)
))

# Increase dropout
model.add(keras.layers.Dropout(0.5))
```

### 2. Underfitting
```python
# Increase model capacity
model = keras.Sequential([
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Adjust learning rate
optimizer = keras.optimizers.Adam(learning_rate=1e-4)
```

### 3. Poor Convergence
```python
# Implement gradient clipping
optimizer = keras.optimizers.Adam(
    learning_rate=1e-3,
    clipnorm=1.0
)

# Use different optimizer
optimizer = keras.optimizers.RMSprop(
    learning_rate=1e-3,
    rho=0.9
)
```

## Further Reading
- [Keras Documentation](https://keras.io/guides/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python) 
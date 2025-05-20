import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
import os
import pickle

def create_model():
    """Create a simple CNN model for digit recognition."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def train_model(existing_model=None):
    """Train the model on MNIST dataset."""
    # Load and preprocess MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    # Normalize and reshape images
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    
    # Create or use existing model
    model = existing_model if existing_model else create_model()
    
    # Train model
    model.fit(train_images, train_labels, epochs=5,
              validation_data=(test_images, test_labels))
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'\nTest accuracy: {test_acc}')
    
    return model

def fine_tune_model(model, images, labels, epochs=1):
    """Fine-tune the model with new examples."""
    if len(images) == 0:
        return model
        
    # Reshape and normalize images
    images = np.array(images).reshape((-1, 28, 28, 1)).astype('float32') / 255
    labels = np.array(labels)
    
    # Fine-tune with a small learning rate
    original_lr = model.optimizer.learning_rate.numpy()
    model.optimizer.learning_rate.assign(original_lr * 0.1)
    
    # Train for a few epochs
    model.fit(images, labels, epochs=epochs, verbose=0)
    
    # Restore original learning rate
    model.optimizer.learning_rate.assign(original_lr)
    
    return model

def save_model_state(model, filename='digit_model.h5'):
    """Save the model weights and optimizer state."""
    model.save_weights(filename)
    print(f"Model state saved to {filename}")

def load_model_state(filename='digit_model.h5'):
    """Load a model with weights and optimizer state."""
    model = create_model()
    if os.path.exists(filename):
        model.load_weights(filename)
        print(f"Loaded model state from {filename}")
    return model

if __name__ == "__main__":
    print("Training digit recognition model...")
    
    # Try to load existing model, or create new one
    model = load_model_state()
    
    # Train or fine-tune the model
    model = train_model(model)
    
    # Save the model state
    save_model_state(model)
    
    print("\nDone! The model is ready to use with the bank scanner application.") 
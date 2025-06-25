import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from PIL import Image
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class SkinDiseaseDetector:
    def __init__(self, train_path='dataset/train', test_path='dataset/test', img_size=(224, 224)):
        self.train_path = train_path
        self.test_path = test_path
        self.img_size = img_size
        self.num_classes = 0
        self.class_names = []
        self.model = None
        self.history = None
        
    def get_class_names(self):
        """Get all class names from the dataset"""
        self.class_names = sorted([d for d in os.listdir(self.train_path) 
                                  if os.path.isdir(os.path.join(self.train_path, d))])
        self.num_classes = len(self.class_names)
        print(f"Found {self.num_classes} classes: {self.class_names}")
        return self.class_names
    
    def create_data_generators(self, batch_size=32):
        """Create data generators for training and validation"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Training generator
        self.train_generator = train_datagen.flow_from_directory(
            self.train_path,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation generator
        self.val_generator = val_datagen.flow_from_directory(
            self.train_path,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Test generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        self.test_generator = test_datagen.flow_from_directory(
            self.test_path,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.val_generator.samples}")
        print(f"Test samples: {self.test_generator.samples}")
        
    def build_model(self):
        """Build the CNN model using transfer learning with ResNet50V2"""
        # Load pre-trained ResNet50V2 model
        base_model = ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Create the model
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model Summary:")
        self.model.summary()
        
    def train_model(self, epochs=50):
        """Train the model with callbacks"""
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            'best_skin_disease_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Train the model
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=[early_stopping, reduce_lr, checkpoint],
            verbose=1
        )
        
        # Fine-tuning: Unfreeze some layers and train with lower learning rate
        print("\nStarting fine-tuning...")
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Freeze the first 100 layers
        for layer in base_model.layers[:100]:
            layer.trainable = False
            
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Continue training
        self.history_fine = self.model.fit(
            self.train_generator,
            epochs=20,
            validation_data=self.val_generator,
            callbacks=[early_stopping, reduce_lr, checkpoint],
            verbose=1
        )
        
    def evaluate_model(self):
        """Evaluate the model on test data"""
        print("\nEvaluating model on test data...")
        test_loss, test_accuracy = self.model.evaluate(self.test_generator, verbose=1)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Predictions
        self.test_generator.reset()
        predictions = self.model.predict(self.test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.test_generator.classes
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        return test_accuracy, y_pred, y_true
        
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        if 'lr' in self.history.history:
            axes[1, 0].plot(self.history.history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def predict_single_image(self, image_path):
        """Predict a single image"""
        # Load and preprocess image
        img = Image.open(image_path)
        img = img.resize(self.img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = self.model.predict(img_array)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        return self.class_names[predicted_class], confidence, prediction[0]
        
    def save_model(self, model_path='skin_disease_model.h5'):
        """Save the trained model"""
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        
    def load_model(self, model_path='skin_disease_model.h5'):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")

def main():
    """Main function to train and evaluate the model"""
    print("=== Skin Disease Detection Model ===")
    
    # Initialize the detector
    detector = SkinDiseaseDetector()
    
    # Get class names
    detector.get_class_names()
    
    # Create data generators
    detector.create_data_generators(batch_size=32)
    
    # Build model
    detector.build_model()
    
    # Train model
    detector.train_model(epochs=30)
    
    # Evaluate model
    test_accuracy, y_pred, y_true = detector.evaluate_model()
    
    # Plot results
    detector.plot_training_history()
    detector.plot_confusion_matrix(y_true, y_pred)
    
    # Save model
    detector.save_model()
    
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 
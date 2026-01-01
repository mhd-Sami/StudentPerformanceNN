"""
Deep Learning Model Training
Implements neural network models for student performance prediction.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

from feature_engineering import FeatureEngineer


class StudentPerformanceNN:
    """Neural network models for student performance prediction."""
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.models = {}
        self.history = {}
        
    def create_classification_model(self, name='classifier'):
        """
        Create a neural network for binary classification.
        
        Architecture:
        - Input layer
        - 3 hidden layers with batch normalization and dropout
        - Output layer with sigmoid activation
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.input_dim,)),
            
            # First hidden layer
            layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            
            # Second hidden layer
            layers.Dense(64, kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.25),
            
            # Third hidden layer
            layers.Dense(32, kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ], name=name)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def create_regression_model(self, name='regressor'):
        """
        Create a neural network for regression.
        
        Architecture:
        - Input layer
        - 3 hidden layers with batch normalization and dropout
        - Output layer with linear activation
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.input_dim,)),
            
            # First hidden layer
            layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            
            # Second hidden layer
            layers.Dense(64, kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.25),
            
            # Third hidden layer
            layers.Dense(32, kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(1, activation='linear')
        ], name=name)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', keras.metrics.RootMeanSquaredError(name='rmse')]
        )
        
        return model
    
    def create_attention_model(self, name='attention_classifier'):
        """
        Create an attention-based neural network.
        
        Uses self-attention mechanism to focus on important features.
        """
        inputs = layers.Input(shape=(self.input_dim,))
        
        # Feature embedding
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        
        # Self-attention mechanism
        attention_weights = layers.Dense(64, activation='softmax', name='attention')(x)
        attended_features = layers.Multiply()([x, attention_weights])
        
        # Further processing
        x = layers.Dense(32, activation='relu')(attended_features)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Output
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name=name)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def get_callbacks(self, model_name):
        """Create training callbacks."""
        return [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            
            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath=f'saved_models/{model_name}_best.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
    
    def train_model(self, model, X_train, y_train, X_val, y_val, 
                   model_name, epochs=100, batch_size=32):
        """
        Train a neural network model.
        
        Args:
            model: Keras model
            X_train, y_train: Training data
            X_val, y_val: Validation data
            model_name: Name for saving
            epochs: Maximum epochs
            batch_size: Batch size
            
        Returns:
            Trained model and history
        """
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(model_name),
            verbose=1
        )
        
        self.models[model_name] = model
        self.history[model_name] = history
        
        return model, history
    
    def evaluate_classifier(self, model, X_test, y_test, model_name):
        """Evaluate classification model."""
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")
        
        # Predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nAccuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Fail', 'Pass']))
        
        return accuracy, y_pred
    
    def evaluate_regressor(self, model, X_test, y_test, model_name):
        """Evaluate regression model."""
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")
        
        # Predictions
        y_pred = model.predict(X_test, verbose=0).flatten()
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        
        print(f"\nRoot Mean Squared Error: {rmse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        return rmse, r2, y_pred
    
    def plot_training_history(self, model_name, save_path=None):
        """Plot training history."""
        if model_name not in self.history:
            print(f"No history found for {model_name}")
            return
        
        history = self.history[model_name].history
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(history['loss'], label='Training Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{model_name} - Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Metric plot
        metric_key = 'accuracy' if 'accuracy' in history else 'mae'
        val_metric_key = f'val_{metric_key}'
        
        axes[1].plot(history[metric_key], label=f'Training {metric_key}')
        axes[1].plot(history[val_metric_key], label=f'Validation {metric_key}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_key.upper())
        axes[1].set_title(f'{model_name} - {metric_key.upper()}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


def main():
    """Main training pipeline for neural networks."""
    print("="*60)
    print("Student Performance Prediction - Neural Network Training")
    print("="*60)
    
    # Load data
    print("\nLoading training data...")
    data_path = Path(__file__).parent.parent / 'data' / 'training_data.csv'
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} records")
    
    # Feature engineering
    print("\nEngineering features...")
    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_features(df, include_poly=False)
    
    # Prepare features
    feature_columns = [col for col in df_engineered.columns 
                      if col not in ['pass_fail', 'final_exam', 'needs_support']]
    
    X = df_engineered[feature_columns].values
    print(f"✓ Created {X.shape[1]} features")
    
    # Scale features
    X_train_full, X_test = train_test_split(X, test_size=0.2, random_state=42)
    X_train, X_val = train_test_split(X_train_full, test_size=0.2, random_state=42)
    
    X_train_scaled, X_val_scaled = engineer.scale_features(X_train, X_val)
    X_test_scaled = engineer.scaler.transform(X_test)
    
    # Prepare targets
    y_pass_fail = df['pass_fail'].values
    y_final_score = df['final_exam'].values
    y_support = df['needs_support'].values
    
    y_pf_train, y_pf_test = train_test_split(y_pass_fail, test_size=0.2, random_state=42)
    y_pf_train, y_pf_val = train_test_split(y_pf_train, test_size=0.2, random_state=42)
    
    y_fs_train, y_fs_test = train_test_split(y_final_score, test_size=0.2, random_state=42)
    y_fs_train, y_fs_val = train_test_split(y_fs_train, test_size=0.2, random_state=42)
    
    y_sp_train, y_sp_test = train_test_split(y_support, test_size=0.2, random_state=42)
    y_sp_train, y_sp_val = train_test_split(y_sp_train, test_size=0.2, random_state=42)
    
    # Initialize neural network trainer
    nn_trainer = StudentPerformanceNN(input_dim=X_train_scaled.shape[1])
    
    # Train Pass/Fail Classifier
    pf_model = nn_trainer.create_classification_model('pass_fail_nn')
    nn_trainer.train_model(
        pf_model, X_train_scaled, y_pf_train, X_val_scaled, y_pf_val,
        'pass_fail_nn', epochs=100, batch_size=32
    )
    nn_trainer.evaluate_classifier(pf_model, X_test_scaled, y_pf_test, 'Pass/Fail NN')
    
    # Train Final Score Regressor
    fs_model = nn_trainer.create_regression_model('final_score_nn')
    nn_trainer.train_model(
        fs_model, X_train_scaled, y_fs_train, X_val_scaled, y_fs_val,
        'final_score_nn', epochs=100, batch_size=32
    )
    nn_trainer.evaluate_regressor(fs_model, X_test_scaled, y_fs_test, 'Final Score NN')
    
    # Train Support Need Classifier
    sp_model = nn_trainer.create_classification_model('support_nn')
    nn_trainer.train_model(
        sp_model, X_train_scaled, y_sp_train, X_val_scaled, y_sp_val,
        'support_nn', epochs=100, batch_size=32
    )
    nn_trainer.evaluate_classifier(sp_model, X_test_scaled, y_sp_test, 'Support Need NN')
    
    # Save models
    models_dir = Path(__file__).parent / 'saved_models'
    models_dir.mkdir(exist_ok=True)
    
    pf_model.save(models_dir / 'pass_fail_nn.h5')
    fs_model.save(models_dir / 'final_score_nn.h5')
    sp_model.save(models_dir / 'support_nn.h5')
    
    # Save scaler and feature names
    joblib.dump(engineer.scaler, models_dir / 'scaler.pkl')
    joblib.dump(feature_columns, models_dir / 'feature_names_nn.pkl')
    
    print("\n" + "="*60)
    print("Neural Network Models Saved Successfully")
    print("="*60)
    print(f"Location: {models_dir}")
    print("Files:")
    print("  - pass_fail_nn.h5")
    print("  - final_score_nn.h5")
    print("  - support_nn.h5")
    print("  - scaler.pkl")
    print("  - feature_names_nn.pkl")
    
    print("\n✓ Training Complete!")


if __name__ == "__main__":
    main()

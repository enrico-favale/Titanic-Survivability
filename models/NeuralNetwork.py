import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import warnings
warnings.filterwarnings('ignore')

class TitanicSurvivalModel:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        self.history = None

    def load_and_preprocess_data(self):
        df = pd.read_csv(self.csv_path)

        # Manteniamo solo le colonne utili
        df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

        # Conversione di "Yes"/"No" in 1/0
        df['Survived'] = df['Survived'].map({'Yes': 1, 'No': 0})

        # Gestione dei valori mancanti
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

        # Encoding delle variabili categoriche
        for column in ['Sex', 'Embarked']:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            self.label_encoders[column] = le
            
        # Drop delle colonne non utili
        columns_to_drop = ['PassengerId', 'Name', 'Ticket']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

        # Separazione tra features e target
        X = df.drop('Survived', axis=1)
        y = df['Survived']

        # Normalizzazione
        X_scaled = self.scaler.fit_transform(X)

        # Split train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

    def build_model(self):
        self.model = tf.keras.models.Sequential([
            # Input layer
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.X_train.shape[1],)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            # Hidden layers
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),
            
            # Output layer
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision')]
        )

    def train(self, epochs=500, batch_size=25, validation_split=0.2):
        self.history = self.model.fit(
            self.X_train, 
            self.y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split
        )

    def evaluate(self):
        loss, accuracy, precision = self.model.evaluate(self.X_test, self.y_test)
        
        print(f'\n\nTest Accuracy: {accuracy:.2f}')
        print(f'Test Precision: {precision:.2f}')
        print(f'Test Loss: {loss:.2f}')
        
        return accuracy, precision
    
    def plot_metrics(self, metrics = ['accuracy', 'precision']):
        plt.figure(figsize=(10, 6))
            
        if 'accuracy' in metrics and self.history.history['accuracy']:
            plt.plot(self.history.history['accuracy'], label='Accuracy')
        if 'precision' in metrics and self.history.history['precision']:
            plt.plot(self.history.history['precision'], label='Precision')

        plt.xlabel('Epochs')
        plt.ylabel('Metric Value')
        plt.title('Evaluation Metrics over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        
        plt.plot(self.history.history['loss'], label='Loss')
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        plt.title('Evaluation Loss over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    def predict(self, passenger_data):
        scaled = self.scaler.transform([passenger_data])
        prob = self.model.predict(scaled)[0][0]
        return {"probability": prob, "prediction": int(prob > 0.5)}

def main():
    model = TitanicSurvivalModel('data/dataset.csv')
    model.load_and_preprocess_data()
    model.build_model()
    model.train()
    model.evaluate()
    model.plot_metrics()
    model.plot_loss()

    # Esempio di passeggero
    sample_passenger = [3, 1, 22.0, 1, 0, 7.25, 2]
    print(model.predict(sample_passenger))
    
if __name__=="__main__":
    main()

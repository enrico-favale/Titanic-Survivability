import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import joblib

import warnings
warnings.filterwarnings('ignore')

class TitanicSurvivalModel:
    def __init__(
        self, 
        dataset_path: str,
        epochs = 500,
        batch_size = 64,
        test_split = 0.1,
        validation_split = 0.2,
        learning_rate = 1e-3,
    ):
        
        self.dataset_path = dataset_path
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_size = test_split
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_and_preprocess_data(self):
        df = pd.read_csv(self.dataset_path)

        # Preprocessing
        df_processed = df.copy()
        
        # 1. Estrai il titolo dal nome prima di rimuovere le colonne
        if 'Name' in df.columns:
            df_processed['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
            df_processed['Title'] = df_processed['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                                        'Don', 'Dr', 'Major', 'Rev', 'Sir', 
                                                                        'Jonkheer', 'Dona'], 'Rare')
            df_processed['Title'] = df_processed['Title'].replace(['Mlle', 'Ms'], 'Miss')
            df_processed['Title'] = df_processed['Title'].replace('Mme', 'Mrs')
        
        # 3. Gestisci i valori mancanti
        # Age: riempi con la mediana
        df_processed['Age'].fillna(df_processed['Age'].median(), inplace=True)
        
        # Embarked: riempi con il valore più frequente
        df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0], inplace=True)
        
        # Cabin: crea una feature binaria per indicare se la cabina è presente
        df_processed['Has_Cabin'] = df_processed['Cabin'].notna().astype(int)
        df_processed = df_processed.drop('Cabin', axis=1)
        
        # 4. Feature engineering aggiuntive
        # Crea feature famiglia
        df_processed['Family_Size'] = df_processed['SibSp'] + df_processed['Parch'] + 1
        df_processed['Is_Alone'] = (df_processed['Family_Size'] == 1).astype(int)
        
        # Crea fasce di età
        df_processed['Age_Group'] = pd.cut(df_processed['Age'], 
                                         bins=[0, 12, 18, 35, 60, 100], 
                                         labels=[0, 1, 2, 3, 4])
        df_processed['Age_Group'] = df_processed['Age_Group'].astype(int)
        
        # Crea fasce di tariffa
        df_processed['Fare_Group'] = pd.qcut(df_processed['Fare'], 
                                           q=4, 
                                           labels=[0, 1, 2, 3])
        df_processed['Fare_Group'] = df_processed['Fare_Group'].astype(int)
        
        # 2. Rimuovi colonne non utili per la predizione
        columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns])
        
        # 5. Converti la variabile target in formato binario
        df_processed['Survived'] = (df_processed['Survived'] == 'Yes').astype(int)
        
        # 6. Encoding delle variabili categoriche
        categorical_columns = ['Sex', 'Embarked']
        if 'Title' in df_processed.columns:
            categorical_columns.append('Title')
            
        for col in categorical_columns:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le

        # Separazione tra features e target
        X = df_processed.drop('Survived', axis=1)
        y = df_processed['Survived']

        # Normalizzazione
        X_scaled = self.scaler.fit_transform(X)

        # Split train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=self.test_size, random_state=42
        )
        
        print("\n✅ Preprocessing completato!")
        print(f"Features finali: {list(df_processed.columns)}")

    def build_model(self):
        self.model = tf.keras.models.Sequential([
            # Input layer
            tf.keras.layers.Dense(160, activation='relu', input_shape=(self.X_train.shape[1],)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),
            
            # Hidden layers
            tf.keras.layers.Dense(104, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            
            tf.keras.layers.Dense(48, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            
            tf.keras.layers.Dense(112, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            # Output layer
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = tf.keras.losses.BinaryCrossentropy()
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                # tf.keras.metrics.Accuracy(name="accuracy"),
                'accuracy',
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.F1Score(name="f1_score")           
            ]
        )

    def train(self):
        
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        
        class_weights_dict = dict(enumerate(class_weights))
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        self.history = self.model.fit(
            self.X_train, 
            self.y_train, 
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            validation_split=self.validation_split,
            callbacks=[early_stopping],
            class_weight=class_weights_dict
        )

    def evaluate(self):
        loss, accuracy, precision, f1_score = self.model.evaluate(self.X_test, self.y_test)
        
        print(f'Test Accuracy: {accuracy:.2f}')
        print(f'Test Precision: {precision:.2f}')
        print(f'Test F1 Score: {f1_score:.2f}')
        print(f'Test Loss: {loss:.2f}')
        
        return loss, accuracy, precision, f1_score
    
    def plot_metrics(self, metrics=['accuracy', 'precision', 'f1_score']):
        for metric in metrics:
            if metric in self.history.history and not metric.startswith('val_'):
                plt.figure(figsize=(10, 6))
                
                plt.plot(self.history.history[metric], label=metric.capitalize())
                if self.history.history[f'val_{metric}']: plt.plot(self.history.history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
                
                plt.xlabel('Epochs')
                plt.ylabel('Metric Value')
                plt.title(f'Evaluation {metric} over Time')
                plt.legend()
                plt.grid(True)
                plt.show()
            else:
                print(f"Metric '{metric}' not found in history.")
                
    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        
        plt.plot(self.history.history['loss'], label='Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        plt.title('Evaluation Loss over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_confusion_matrix(self):
        y_pred = (self.model.predict(self.X_test) > 0.5).astype(int)
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
        
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()
        
    def start(self):
        self.load_and_preprocess_data()
        self.build_model()
        self.train()
        self.evaluate()
        
    def plot_all(self):
        self.plot_metrics()
        self.plot_loss()
        self.plot_confusion_matrix()
        
    def summarize(self):
        print(self.model.summary())

        # Calcolo del numero di esempi
        total_samples = self.X_train.shape[0] + self.X_test.shape[0]
        train_samples = int(self.X_train.shape[0] * (1 - self.validation_split))
        val_samples = int(self.X_train.shape[0] * self.validation_split)
        test_samples = self.X_test.shape[0]

        print(
            f'\nIperparametri:\n\n'
            f' Epochs: {self.epochs}\n'
            f' Batch Size: {self.batch_size}\n'
            f' Learning Rate: {self.learning_rate}\n'
            f'\nDataset Splits:\n\n'
            f' Training Split: {1 - self.test_size:.2f} -> {train_samples} esempi\n'
            f' Validation Split: {self.validation_split:.2f} -> {val_samples} esempi\n'
            f' Test Split: {self.test_size:.2f} -> {test_samples} esempi\n'
            f' Totale: {total_samples} esempi\n'
        )
    def save_model(self, path):
        self.model.save(filepath=f"{path}/model.keras")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        joblib.dump(self.label_encoders, f"{path}/label_encoders.pkl")

def main():
    pass
    
if __name__=="__main__":
    main()

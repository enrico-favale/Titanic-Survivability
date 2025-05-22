import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

class TitanicNeuralNetwork:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        
    def load_and_preprocess_data(self, file_path):
        """
        Carica e preprocessa il dataset del Titanic
        """
        print("📊 Caricamento e preprocessing del dataset...")
        
        # Carica il dataset
        df = pd.read_csv(file_path)
        print(f"Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne")
        
        # Visualizza informazioni base
        print("\n🔍 Informazioni sui valori mancanti:")
        print(df.isnull().sum())
        
        # Preprocessing
        df_processed = df.copy()
        
        # 1. Rimuovi colonne non utili per la predizione
        columns_to_drop = ['PassengerId', 'Name', 'Ticket']
        df_processed = df_processed.drop(columns=columns_to_drop)
        
        # 2. Gestisci i valori mancanti
        # Age: riempi con la mediana
        df_processed['Age'].fillna(df_processed['Age'].median(), inplace=True)
        
        # Embarked: riempi con il valore più frequente
        df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0], inplace=True)
        
        # Cabin: crea una feature binaria per indicare se la cabina è presente
        df_processed['Has_Cabin'] = df_processed['Cabin'].notna().astype(int)
        df_processed = df_processed.drop('Cabin', axis=1)
        
        # 3. Feature engineering
        # Estrai il titolo dal nome (se ancora presente)
        if 'Name' in df.columns:
            df_processed['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
            df_processed['Title'] = df_processed['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
            df_processed['Title'] = df_processed['Title'].replace('Mlle', 'Miss')
            df_processed['Title'] = df_processed['Title'].replace('Ms', 'Miss')
            df_processed['Title'] = df_processed['Title'].replace('Mme', 'Mrs')
        
        # Crea feature famiglia
        df_processed['Family_Size'] = df_processed['SibSp'] + df_processed['Parch'] + 1
        df_processed['Is_Alone'] = (df_processed['Family_Size'] == 1).astype(int)
        
        # 4. Converti la variabile target in formato binario
        df_processed['Survived'] = (df_processed['Survived'] == 'Yes').astype(int)
        
        # 5. Encoding delle variabili categoriche
        categorical_columns = ['Sex', 'Embarked']
        if 'Title' in df_processed.columns:
            categorical_columns.append('Title')
            
        for col in categorical_columns:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
        
        print("\n✅ Preprocessing completato!")
        print(f"Features finali: {list(df_processed.columns)}")
        
        return df_processed
    
    def explore_data(self, df):
        """
        Esplora e visualizza il dataset
        """
        print("\n📈 Analisi esplorativa dei dati...")
        
        # Statistiche descrittive
        print("\nStatistiche descrittive:")
        print(df.describe())
        
        # Correlazione con la sopravvivenza
        print("\n🎯 Correlazione con la sopravvivenza:")
        correlations = df.corr()['Survived'].sort_values(ascending=False)
        print(correlations)
        
        # Visualizzazioni
        plt.figure(figsize=(15, 10))
        
        # Distribuzione sopravvivenza
        plt.subplot(2, 3, 1)
        df['Survived'].value_counts().plot(kind='bar', color=['red', 'green'])
        plt.title('Distribuzione Sopravvivenza')
        plt.xlabel('Sopravvissuto (0=No, 1=Sì)')
        plt.ylabel('Numero passeggeri')
        
        # Sopravvivenza per classe
        plt.subplot(2, 3, 2)
        survival_by_class = df.groupby('Pclass')['Survived'].mean()
        survival_by_class.plot(kind='bar', color='skyblue')
        plt.title('Tasso di sopravvivenza per classe')
        plt.xlabel('Classe')
        plt.ylabel('Tasso sopravvivenza')
        
        # Sopravvivenza per sesso
        plt.subplot(2, 3, 3)
        survival_by_sex = df.groupby('Sex')['Survived'].mean()
        survival_by_sex.plot(kind='bar', color='lightcoral')
        plt.title('Tasso di sopravvivenza per sesso')
        plt.xlabel('Sesso (0=Female, 1=Male)')
        plt.ylabel('Tasso sopravvivenza')
        
        # Distribuzione età
        plt.subplot(2, 3, 4)
        plt.hist(df['Age'], bins=30, alpha=0.7, color='gold')
        plt.title('Distribuzione Età')
        plt.xlabel('Età')
        plt.ylabel('Frequenza')
        
        # Heatmap correlazioni
        plt.subplot(2, 3, 5)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Matrice di Correlazione')
        
        plt.tight_layout()
        plt.show()
    
    def prepare_features(self, df):
        """
        Prepara le features per l'addestramento
        """
        # Separa features e target
        X = df.drop('Survived', axis=1)
        y = df['Survived']
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalizzazione
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\n📋 Dati preparati:")
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        print(f"Features utilizzate: {list(X.columns)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def build_model(self, input_dim):
        """
        Costruisce la rete neurale
        """
        print("\n🧠 Costruzione della rete neurale...")
        
        model = keras.Sequential([
            # Layer di input
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Primi layer nascosti
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Secondo layer nascosto
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Terzo layer nascosto
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            
            # Layer di output
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compila il modello
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("✅ Modello costruito!")
        print(model.summary())
        
        return model
    
    def train_model(self, X_train, X_test, y_train, y_test, epochs=100):
        """
        Addestra la rete neurale
        """
        print(f"\n🚀 Addestramento della rete neurale per {epochs} epoche...")
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        # Addestramento
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("✅ Addestramento completato!")
        return history
    
    def evaluate_model(self, X_test, y_test):
        """
        Valuta le performance del modello
        """
        print("\n📊 Valutazione del modello...")
        
        # Predizioni
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Metriche
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n🎯 Accuracy: {accuracy:.4f}")
        
        print("\n📈 Report di classificazione:")
        print(classification_report(y_test, y_pred))
        
        # Matrice di confusione
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matrice di Confusione')
        plt.xlabel('Predetto')
        plt.ylabel('Reale')
        plt.show()
        
        return accuracy, y_pred_prob
    
    def plot_training_history(self, history):
        """
        Visualizza l'andamento dell'addestramento
        """
        plt.figure(figsize=(15, 5))
        
        # Loss
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Andamento Loss')
        plt.xlabel('Epoca')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy
        plt.subplot(1, 3, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Andamento Accuracy')
        plt.xlabel('Epoca')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Learning Rate (se disponibile)
        if 'lr' in history.history:
            plt.subplot(1, 3, 3)
            plt.plot(history.history['lr'])
            plt.title('Learning Rate')
            plt.xlabel('Epoca')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def predict_passenger(self, passenger_data):
        """
        Predice la sopravvivenza per un singolo passeggero
        """
        # Preprocessa i dati del passeggero
        passenger_df = pd.DataFrame([passenger_data])
        
        # Applica gli stessi encoding utilizzati durante l'addestramento
        for col, encoder in self.label_encoders.items():
            if col in passenger_df.columns:
                passenger_df[col] = encoder.transform(passenger_df[col])
        
        # Normalizza
        passenger_scaled = self.scaler.transform(passenger_df)
        
        # Predizione
        prob = self.model.predict(passenger_scaled)[0][0]
        prediction = "Sopravvissuto" if prob > 0.5 else "Non sopravvissuto"
        
        print(f"\n🔮 Predizione per il passeggero:")
        print(f"Probabilità di sopravvivenza: {prob:.4f}")
        print(f"Predizione: {prediction}")
        
        return prob, prediction

# Esempio di utilizzo
def main():
    # Inizializza il modello
    titanic_nn = TitanicNeuralNetwork()
    
    # Carica e preprocessa i dati
    df = titanic_nn.load_and_preprocess_data('6_titanic.csv')
    
    # Esplora i dati
    titanic_nn.explore_data(df)
    
    # Prepara le features
    X_train, X_test, y_train, y_test = titanic_nn.prepare_features(df)
    
    # Costruisce il modello
    titanic_nn.model = titanic_nn.build_model(X_train.shape[1])
    
    # Addestra il modello
    history = titanic_nn.train_model(X_train, X_test, y_train, y_test, epochs=100)
    
    # Visualizza l'andamento dell'addestramento
    titanic_nn.plot_training_history(history)
    
    # Valuta il modello
    accuracy, predictions = titanic_nn.evaluate_model(X_test, y_test)
    
    # Esempio di predizione per un nuovo passeggero
    new_passenger = {
        'Pclass': 1,           # Prima classe
        'Sex': 'female',       # Femmina
        'Age': 25,             # 25 anni
        'SibSp': 0,            # Nessun fratello/sorella a bordo
        'Parch': 1,            # Un genitore/figlio a bordo
        'Fare': 80.0,          # Tariffa pagata
        'Embarked': 'S',       # Imbarco a Southampton
        'Has_Cabin': 1,        # Ha una cabina
        'Family_Size': 2,      # Dimensione famiglia
        'Is_Alone': 0          # Non è sola
    }
    
    prob, prediction = titanic_nn.predict_passenger(new_passenger)
    
    print(f"\n🎉 Addestramento completato!")
    print(f"Accuracy finale: {accuracy:.4f}")

if __name__ == "__main__":
    main()

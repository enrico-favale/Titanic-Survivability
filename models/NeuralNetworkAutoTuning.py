import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras_tuner as kt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

class TitanicSurvivalModelAutoTuning:
    def __init__(
        self, 
        dataset_path: str,
        epochs=100,
        batch_size=32,
        test_split=0.1,
        validation_split=0.2,
        max_trials=10,
    ):
        self.dataset_path = dataset_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_size = test_split
        self.validation_split = validation_split
        self.max_trials = max_trials

        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_and_preprocess_data(self):
        df = pd.read_csv(self.dataset_path)
        df_processed = df.copy()

        if 'Name' in df.columns:
            df_processed['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
            df_processed['Title'] = df_processed['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
            df_processed['Title'] = df_processed['Title'].replace('Mlle', 'Miss')
            df_processed['Title'] = df_processed['Title'].replace('Ms', 'Miss')
            df_processed['Title'] = df_processed['Title'].replace('Mme', 'Mrs')

        columns_to_drop = ['PassengerId', 'Name', 'Ticket']
        df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns])

        df_processed['Age'].fillna(df_processed['Age'].median(), inplace=True)
        df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0], inplace=True)
        df_processed['Has_Cabin'] = df_processed['Cabin'].notna().astype(int)
        df_processed = df_processed.drop('Cabin', axis=1)

        df_processed['Family_Size'] = df_processed['SibSp'] + df_processed['Parch'] + 1
        df_processed['Is_Alone'] = (df_processed['Family_Size'] == 1).astype(int)

        df_processed['Age_Group'] = pd.cut(df_processed['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4]).astype(int)
        df_processed['Fare_Group'] = pd.qcut(df_processed['Fare'], q=4, labels=[0, 1, 2, 3]).astype(int)

        df_processed['Survived'] = (df_processed['Survived'] == 'Yes').astype(int)

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

        X = df_processed.drop('Survived', axis=1)
        y = df_processed['Survived']
        X_scaled = self.scaler.fit_transform(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=self.test_size, random_state=42
        )

    def model_builder(self, hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(
            units=hp.Int('units_input', min_value=8, max_value=256, step=8),
            activation=hp.Choice('activation_input', ['relu']),
            input_shape=(self.X_train.shape[1],)
        ))
        for i in range(hp.Int("n_layers", 1, 3)):
            model.add(tf.keras.layers.Dense(
                units=hp.Int(f'units_{i}', min_value=8, max_value=128, step=8),
                activation=hp.Choice(f'activation_{i}', ['relu'])
            ))
            model.add(tf.keras.layers.Dropout(
                rate=hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)
            ))

        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.Float('lr', 1e-4, 1e-2, sampling='log')
            ),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.F1Score()]
        )
        return model

    def tune_model(self):
        tuner = kt.RandomSearch(
            self.model_builder,
            objective=kt.Objective('val_accuracy', direction='max'),
            max_trials=self.max_trials,
            executions_per_trial=2,
            directory='keras_tuner',
            project_name='titanic_tuning'
        )

        tuner.search(self.X_train, self.y_train,
                     epochs=self.epochs,
                     validation_split=self.validation_split,
                     callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])

        self.best_hp = tuner.get_best_hyperparameters(1)[0]
        self.model = tuner.get_best_models(1)[0]
        print("\n✅ Migliori iperparametri trovati:")
        for param in self.best_hp.values.keys():
            print(f"{param}: {self.best_hp.get(param)}")

    def train_best_model(self):
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

    def plot_metrics(self, metrics=['accuracy', 'precision']):
        plt.figure(figsize=(10, 6))
        if 'accuracy' in metrics and 'accuracy' in self.history.history:
            plt.plot(self.history.history['accuracy'], label='Accuracy')
        if 'precision' in metrics and 'precision' in self.history.history:
            plt.plot(self.history.history['precision'], label='Precision')
        if 'f1_score' in metrics and self.history.history['f1_score']:
            plt.plot(self.history.history['f1_score'], label='F1 Score')

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

    def plot_confusion_matrix(self):
        y_pred = (self.model.predict(self.X_test) > 0.5).astype(int)
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

    def predict(self, passenger_data):
        scaled = self.scaler.transform([passenger_data])
        prob = self.model.predict(scaled)[0][0]
        return {"probability": prob, "prediction": int(prob > 0.5)}

def main():
    model = TitanicSurvivalModelAutoTuning('data/dataset.csv')
    model.load_and_preprocess_data()
    model.tune_model()
    model.train_best_model()
    model.evaluate()
    model.plot_metrics()
    model.plot_loss()

if __name__ == "__main__":
    main()

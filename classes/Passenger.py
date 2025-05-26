import tensorflow as tf
import joblib
import random
import pandas as pd
import numpy as np

class Passenger:
    def __init__(
        self,
        Pclass: int,
        Sex: str,
        Title: str,
        Age: float,
        SibSp: int,
        Parch: int,
        Fare: float,
        Embarked: str,
    ):
        
        self.scaler = joblib.load("./saved_models/scaler.pkl")
        self.model = tf.keras.models.load_model("./saved_models/model.keras")
        self.label_encoders = joblib.load("./saved_models/label_encoders.pkl")
        
        assert Pclass in [1, 2, 3], "Pclass must be 1, 2, or 3"
        assert Sex in ["male", "female"], "Sex must be male or female"
        assert Title in ["Rare", "Miss", "Mr"], "Title must be Rare, Miss or Mr"
        assert Age >= 0, "Age must be greater than 0"
        assert Fare >= 0, "Fare must be greater than 0"
        assert Embarked in ["C", "Q", "S"], "Embarked must be C, Q, or S"
        
        self.Pclass = Pclass
        self.Sex = Sex
        self.Title = Title
        self.Age = Age
        self.SibSp = SibSp
        self.Parch = Parch
        self.Fare = Fare
        self.Embarked = Embarked
        
        self.Has_Cabin = int(random.random() > 0.8)
        self.Family_Size = self.SibSp + self.Parch + 1
        self.Is_Alone = int(self.Family_Size == 1)
        self.Age_Group = (pd.cut(x=[self.Age], 
                                bins=[0, 12, 18, 35, 60, 100], 
                                labels=[0, 1, 2, 3, 4])).astype(int)
        self.Fare_Group = (pd.cut(x=[self.Fare], 
                                bins=[0.0, 7.9104, 14.4542, 31.0, 512.3292], 
                                labels=[0, 1, 2, 3])).astype(int)
        
        self.Sex = int(self.label_encoders['Sex'].transform([self.Sex])[0])
        self.Title = int(self.label_encoders['Title'].transform([self.Title])[0])
        self.Embarked = int(self.label_encoders['Embarked'].transform([self.Embarked])[0])
        
    def predict(self):
        passenger_array = {
            'Pclass': [self.Pclass],
            'Sex': [self.Sex],
            'Age': [self.Age],
            'SibSp': [self.SibSp],
            'Parch': [self.Parch],
            'Fare': [self.Fare],
            'Embarked': [self.Embarked],
            'Title': [self.Title],
            'Has_Cabin': [self.Has_Cabin],
            'Family_Size': [self.Family_Size],
            'Is_Alone': [self.Is_Alone],
            'Age_Group': [int(self.Age_Group[0])],
            'Fare_Group': [int(self.Fare_Group[0])]
        }

        df = pd.DataFrame(passenger_array)

        scaled = self.scaler.transform(df)
        prob = self.model.predict(scaled)[0][0]

        print(f'Survivability probability: {prob:.2f}\nPrediction: { "Survived" if int(prob > 0.5) else "Not Survived"}')

# 🚢 Titanic Survivability Prediction with TensorFlow

This project uses a neural network built with TensorFlow to predict the survival probability of Titanic passengers, based on a set of engineered features. The model can be used in production through a convenient object-oriented interface via the `Passenger` class.

---

## 📊 Dataset

The dataset used is the well-known [Titanic dataset](https://www.kaggle.com/c/titanic/data) available on Kaggle. It contains demographic and travel information for passengers, including:

* **Pclass**: ticket class (1st, 2nd, 3rd)
* **Sex**: gender
* **Age**: age
* **SibSp**: number of siblings/spouses aboard
* **Parch**: number of parents/children aboard
* **Fare**: ticket fare
* **Embarked**: port of embarkation
* **Cabin**: cabin (partially available)

During preprocessing, the following engineered features were created:

* **Title**: title extracted from the name (e.g., Mr, Miss, Rare)
* **Has\_Cabin**: boolean derived from cabin field availability
* **Family\_Size**: SibSp + Parch + 1
* **Is\_Alone**: boolean, 1 if traveling alone
* **Age\_Group** and **Fare\_Group**: binned versions of Age and Fare

---

## 🧠 Model

The model is a dense neural network (`Dense`) with the following architecture:

```text
Input -> Dense(64, relu) -> Dropout(0.3)
      -> Dense(32, relu) -> Dropout(0.3)
      -> Dense(1, sigmoid)
```

* **Loss**: `binary_crossentropy`
* **Optimizer**: `Adam`
* **Metrics**: `accuracy`, `Precision`, `F1-score`

---

## 🧪 Hyperparameters

* **Epochs**: 37
* **Batch size**: 64
* **Validation split**: 0.2
* **Test split**: 0.1
* **EarlyStopping** to prevent overfitting with a patience of 5

---

## 👻 Using the `Passenger` Class

Once the model is trained, you can predict a passenger's survival by creating a `Passenger` object:

```python
from classes.Passenger import Passenger

passenger = Passenger(
    Pclass=1,
    Sex="male",
    Title="Mr",
    Age=30.0,
    SibSp=0,
    Parch=0,
    Fare=7.25,
    Embarked="S",
)

passenger.predict()
```

The prediction will return both the **survival probability** and the **predicted class** (0 = not survived, 1 = survived).

---

## 🗃️ Saved Files Structure

The model and preprocessing steps are saved under the `saved_models/` directory:

```
saved_models/
│
├── model.keras              # TensorFlow-saved model
├── scaler.pkl              # StandardScaler used for normalization
└── label_encoders.pkl      # Dict with LabelEncoders for categorical columns
```

---

## 📦 Requirements

* Python 3.10+
* TensorFlow 2.15+
* Scikit-learn
* Pandas
* Numpy
* Matplotlib
* Joblib

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 📈 Visualizing Metrics

During training, you can visualize the metrics progression using the `plot_metrics()` function.

---

## ✅ Future To-dos

* Add interactive UI for real-time predictions
* REST API version with Flask or FastAPI
* Integration with a React-based frontend

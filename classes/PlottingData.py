import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder

class PlottingData:
    def __init__(self, dataset: str = "data/dataset.csv"):
        self.dataset_path = dataset
        self.df = None
        self.load_data()

    def load_data(self):
        self.df = pd.read_csv(self.dataset_path)
        print("✅ Dataset caricato con successo!")

    def show_info(self):
        print("\n📋 Informazioni generali:")
        print(self.df.info())
        print("\n🔍 Prime righe del dataset:")
        print(self.df.head())
        
    def preprocessing(self):
        self.label_encoders = {}
        if 'Name' in self.df.columns:
            self.df['Title'] = self.df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
            self.df['Title'] = self.df['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
            self.df['Title'] = self.df['Title'].replace('Mlle', 'Miss')
            self.df['Title'] = self.df['Title'].replace('Ms', 'Miss')
            self.df['Title'] = self.df['Title'].replace('Mme', 'Mrs')
        
        columns_to_drop = ['PassengerId', 'Name', 'Ticket']
        self.df = self.df.drop(columns=[col for col in columns_to_drop if col in self.df.columns])
        
        self.df['Age'].fillna(self.df['Age'].median(), inplace=True)
                
        self.df['Embarked'].fillna(self.df['Embarked'].mode()[0], inplace=True)
        
        self.df['Has_Cabin'] = self.df['Cabin'].notna().astype(int)
        self.df = self.df.drop('Cabin', axis=1)
        
        self.df['Family_Size'] = self.df['SibSp'] + self.df['Parch'] + 1
        self.df['Is_Alone'] = (self.df['Family_Size'] == 1).astype(int)
        
        self.df['Age_Group'] = pd.cut(self.df['Age'], 
                                         bins=[0, 12, 18, 35, 60, 100], 
                                         labels=[0, 1, 2, 3, 4])
        self.df['Age_Group'] = self.df['Age_Group'].astype(int)
        
        self.df['Fare_Group'] = pd.qcut(self.df['Fare'], 
                                           q=4, 
                                           labels=[0, 1, 2, 3])
        self.df['Fare_Group'] = self.df['Fare_Group'].astype(int)
        
        self.df['Survived'] = (self.df['Survived'] == 'Yes').astype(int)
        
        categorical_columns = ['Sex', 'Embarked']
        if 'Title' in self.df.columns:
            categorical_columns.append('Title')
            
        for col in categorical_columns:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
        
        print(self.df['Survived'].value_counts())        
        
        print("\n✅ Preprocessing completato!")
        print(f"Features finali: {list(self.df.columns)}")

    def missing_values(self):
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if not missing.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(x=missing.values, y=missing.index, palette="viridis")
            plt.title("📉 Valori Mancanti per Colonna")
            plt.xlabel("Conteggio")
            plt.ylabel("Colonne")
            plt.grid(True)
            plt.show()
        else:
            print("✅ Nessun valore mancante trovato.")

    def plot_numerical_distributions(self):
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns

        self.df[numerical_cols].hist(
            figsize=(15, 10), bins=20, edgecolor='black', color='skyblue'
        )
        plt.suptitle("📊 Distribuzioni delle Variabili Numeriche", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def plot_categorical_counts(self):
        wanted_cols = ['Survived', 'Sex', 'Age', 'Embarked', 'Pclass']

        categorical_cols = [col for col in wanted_cols if col in self.df.columns]

        for col in categorical_cols:
            plt.figure(figsize=(8, 4))
            sns.countplot(x=col, data=self.df, palette='pastel')
            plt.title(f"📈 Conteggio per categoria: {col}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    def plot_correlation_matrix(self):
        numerical_df = self.df.select_dtypes(include=[np.number])
        corr = numerical_df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("📌 Matrice di Correlazione")
        plt.show()

    def survival_by_feature(self, feature: str):
        if 'Survived' not in self.df.columns:
            print("❌ La colonna 'Survived' non è presente nel dataset.")
            return

        if self.df[feature].dtype == 'object' or self.df[feature].nunique() < 20:
            plt.figure(figsize=(8, 4))
            sns.countplot(x=feature, hue='Survived', data=self.df, palette='Set2')
            plt.title(f"🎯 Sopravvivenza per {feature}")
            plt.xlabel(feature)
            plt.ylabel("Conteggio")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print("⚠️ Feature troppo continua o non adatta per questa visualizzazione.")
if __name__ == "__main__":
    plotter = PlottingData("data/dataset.csv")
    plotter.show_info()
    plotter.missing_values()
    plotter.plot_numerical_distributions()
    plotter.plot_categorical_counts()
    plotter.plot_correlation_matrix()
    plotter.survival_by_feature("Sex")

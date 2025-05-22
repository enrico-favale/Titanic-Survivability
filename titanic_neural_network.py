import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

class TitanicMLPNetwork:
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

        # 1. Estrai il titolo dal nome prima di rimuovere le colonne
        if 'Name' in df.columns:
            df_processed['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
            df_processed['Title'] = df_processed['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
            df_processed['Title'] = df_processed['Title'].replace('Mlle', 'Miss')
            df_processed['Title'] = df_processed['Title'].replace('Ms', 'Miss')
            df_processed['Title'] = df_processed['Title'].replace('Mme', 'Mrs')

        # 2. Rimuovi colonne non utili per la predizione
        columns_to_drop = ['PassengerId', 'Name', 'Ticket']
        df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns])

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
        plt.xticks(rotation=0)

        # Sopravvivenza per classe
        plt.subplot(2, 3, 2)
        survival_by_class = df.groupby('Pclass')['Survived'].mean()
        survival_by_class.plot(kind='bar', color='skyblue')
        plt.title('Tasso di sopravvivenza per classe')
        plt.xlabel('Classe')
        plt.ylabel('Tasso sopravvivenza')
        plt.xticks(rotation=0)

        # Sopravvivenza per sesso
        plt.subplot(2, 3, 3)
        survival_by_sex = df.groupby('Sex')['Survived'].mean()
        survival_by_sex.plot(kind='bar', color='lightcoral')
        plt.title('Tasso di sopravvivenza per sesso')
        plt.xlabel('Sesso (0=Female, 1=Male)')
        plt.ylabel('Tasso sopravvivenza')
        plt.xticks(rotation=0)

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

        # Feature importance dopo l'addestramento di un modello semplice
        plt.subplot(2, 3, 6)
        X_temp = df.drop('Survived', axis=1)
        y_temp = df['Survived']
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_temp, y_temp)
        feature_importance = pd.DataFrame({
            'feature': X_temp.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=True)

        plt.barh(range(len(feature_importance)), feature_importance['importance'])
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.title('Importanza delle Features')
        plt.xlabel('Importanza')

        plt.tight_layout()
        plt.savefig('titanic_data_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Grafico salvato come 'titanic_data_exploration.png'")

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

    def build_and_train_model(self, X_train, X_test, y_train, y_test):
        """
        Costruisce e addestra la rete neurale MLP
        """
        print("\n🧠 Costruzione e addestramento della rete neurale MLP...")

        # Configura il modello MLP
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32, 16),  # 4 layer nascosti
            activation='relu',                      # Funzione di attivazione
            solver='adam',                         # Optimizer
            alpha=0.001,                           # Parametro di regolarizzazione L2
            learning_rate='adaptive',              # Learning rate adattivo
            learning_rate_init=0.001,              # Learning rate iniziale
            max_iter=1000,                         # Numero massimo di iterazioni
            early_stopping=True,                   # Early stopping
            validation_fraction=0.1,               # Frazione per validation
            n_iter_no_change=20,                   # Patience per early stopping
            random_state=42,                       # Seed per riproducibilità
            verbose=True                           # Mostra progress
        )

        print("Parametri del modello:")
        print(f"- Hidden layers: {self.model.hidden_layer_sizes}")
        print(f"- Activation: {self.model.activation}")
        print(f"- Solver: {self.model.solver}")
        print(f"- Alpha (L2): {self.model.alpha}")
        print(f"- Learning rate: {self.model.learning_rate}")

        # Addestramento
        print(f"\n🚀 Addestramento in corso...")
        self.model.fit(X_train, y_train)

        print("✅ Addestramento completato!")
        print(f"Numero di iterazioni: {self.model.n_iter_}")
        print(f"Loss finale: {self.model.loss_:.6f}")

        return self.model

    def evaluate_model(self, X_test, y_test):
        """
        Valuta le performance del modello
        """
        print("\n📊 Valutazione del modello...")

        # Predizioni
        y_pred = self.model.predict(X_test)
        y_pred_prob = self.model.predict_proba(X_test)[:, 1]  # Probabilità classe positiva

        # Metriche
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n🎯 Accuracy: {accuracy:.4f}")

        print("\n📈 Report di classificazione:")
        print(classification_report(y_test, y_pred))

        # Visualizzazioni
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Matrice di confusione
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Matrice di Confusione')
        axes[0,0].set_xlabel('Predetto')
        axes[0,0].set_ylabel('Reale')

        # Distribuzione delle probabilità
        axes[0,1].hist(y_pred_prob[y_test == 0], alpha=0.5, label='Non sopravvissuti', bins=20, color='red')
        axes[0,1].hist(y_pred_prob[y_test == 1], alpha=0.5, label='Sopravvissuti', bins=20, color='green')
        axes[0,1].set_xlabel('Probabilità di sopravvivenza')
        axes[0,1].set_ylabel('Frequenza')
        axes[0,1].set_title('Distribuzione delle Probabilità')
        axes[0,1].legend()

        # Curva di loss durante l'addestramento
        if hasattr(self.model, 'loss_curve_'):
            axes[1,0].plot(self.model.loss_curve_, label='Training Loss')
            axes[1,0].set_xlabel('Iterazioni')
            axes[1,0].set_ylabel('Loss')
            axes[1,0].set_title('Curva di Loss')
            axes[1,0].legend()

            if hasattr(self.model, 'validation_scores_'):
                axes[1,1].plot(self.model.validation_scores_, label='Validation Score')
                axes[1,1].set_xlabel('Iterazioni')
                axes[1,1].set_ylabel('Score')
                axes[1,1].set_title('Score di Validazione')
                axes[1,1].legend()

        plt.tight_layout()
        plt.savefig('titanic_model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Grafico salvato come 'titanic_model_evaluation.png'")

        return accuracy, y_pred_prob

    def plot_learning_curves(self, X, y):
        """
        Visualizza le curve di apprendimento
        """
        print("\n📈 Generazione curve di apprendimento...")

        # Crea un modello semplificato per le curve di apprendimento
        model_simple = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42
        )

        train_sizes, train_scores, val_scores = learning_curve(
            model_simple, X, y, cv=5,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy',
            n_jobs=-1
        )

        plt.figure(figsize=(10, 6))

        # Calcola media e deviazione standard
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Plot delle curve
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')

        plt.xlabel('Dimensione Training Set')
        plt.ylabel('Accuracy')
        plt.title('Curve di Apprendimento')
        plt.legend()
        plt.grid(True)
        plt.savefig('titanic_learning_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Grafico salvato come 'titanic_learning_curves.png'")

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
        prob = self.model.predict_proba(passenger_scaled)[0][1]
        prediction = "Sopravvissuto" if prob > 0.5 else "Non sopravvissuto"

        print(f"\n🔮 Predizione per il passeggero:")
        print(f"Probabilità di sopravvivenza: {prob:.4f}")
        print(f"Predizione: {prediction}")

        return prob, prediction

    def analyze_feature_importance(self, X, y):
        """
        Analizza l'importanza delle features usando diversi metodi
        """
        print("\n🔍 Analisi dell'importanza delle features...")

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.linear_model import LogisticRegression

        feature_names = X.columns if hasattr(X, 'columns') else [f'Feature_{i}' for i in range(X.shape[1])]

        # Random Forest Feature Importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = rf.feature_importances_

        # SelectKBest con f_classif
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        f_scores = selector.scores_

        # Logistic Regression coefficients
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(self.scaler.fit_transform(X), y)
        lr_coeff = np.abs(lr.coef_[0])

        # Crea DataFrame per visualizzazione
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'RF_Importance': rf_importance,
            'F_Score': f_scores,
            'LR_Coefficient': lr_coeff
        })

        # Normalizza i valori per confronto
        importance_df['RF_Importance_Norm'] = importance_df['RF_Importance'] / importance_df['RF_Importance'].max()
        importance_df['F_Score_Norm'] = importance_df['F_Score'] / importance_df['F_Score'].max()
        importance_df['LR_Coefficient_Norm'] = importance_df['LR_Coefficient'] / importance_df['LR_Coefficient'].max()

        # Visualizzazione
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Random Forest
        importance_df.sort_values('RF_Importance', ascending=True).plot(
            x='Feature', y='RF_Importance', kind='barh', ax=axes[0], color='skyblue'
        )
        axes[0].set_title('Random Forest - Feature Importance')
        axes[0].set_xlabel('Importanza')

        # F-Score
        importance_df.sort_values('F_Score', ascending=True).plot(
            x='Feature', y='F_Score', kind='barh', ax=axes[1], color='lightgreen'
        )
        axes[1].set_title('F-Score - Feature Importance')
        axes[1].set_xlabel('F-Score')

        # Logistic Regression
        importance_df.sort_values('LR_Coefficient', ascending=True).plot(
            x='Feature', y='LR_Coefficient', kind='barh', ax=axes[2], color='salmon'
        )
        axes[2].set_title('Logistic Regression - |Coefficienti|')
        axes[2].set_xlabel('|Coefficiente|')

        plt.tight_layout()
        plt.savefig('titanic_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Grafico salvato come 'titanic_feature_importance.png'")

        return importance_df

    def create_final_summary_plot(self, accuracy, y_test, y_pred_prob, importance_df):
        """
        Crea un grafico di riepilogo finale con tutte le metriche principali
        """
        print("\n📊 Creazione grafico di riepilogo finale...")

        fig = plt.figure(figsize=(20, 12))

        # 1. Metriche principali (testo)
        ax1 = plt.subplot(3, 4, 1)
        ax1.axis('off')

        # Calcola metriche aggiuntive
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        y_pred = (y_pred_prob > 0.5).astype(int)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)

        metrics_text = f"""
🎯 METRICHE DEL MODELLO
━━━━━━━━━━━━━━━━━━━━━━━
✅ Accuracy: {accuracy:.3f}
🎪 Precision: {precision:.3f}
🔍 Recall: {recall:.3f}
⚖️ F1-Score: {f1:.3f}
📈 AUC-ROC: {auc:.3f}

🧠 ARCHITETTURA RETE
━━━━━━━━━━━━━━━━━━━━━━━
📊 Hidden Layers: {len(self.model.hidden_layer_sizes)}
🔢 Neuroni: {self.model.hidden_layer_sizes}
🔄 Iterazioni: {self.model.n_iter_}
📉 Loss finale: {self.model.loss_:.6f}
        """

        ax1.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

        # 2. Matrice di confusione
        ax2 = plt.subplot(3, 4, 2)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                   xticklabels=['Non Sopravv.', 'Sopravv.'],
                   yticklabels=['Non Sopravv.', 'Sopravv.'])
        ax2.set_title('Matrice di Confusione', fontsize=14, fontweight='bold')

        # 3. Distribuzione probabilità
        ax3 = plt.subplot(3, 4, 3)
        ax3.hist(y_pred_prob[y_test == 0], alpha=0.6, label='Non sopravvissuti', 
                bins=20, color='red', density=True)
        ax3.hist(y_pred_prob[y_test == 1], alpha=0.6, label='Sopravvissuti', 
                bins=20, color='green', density=True)
        ax3.set_xlabel('Probabilità di sopravvivenza')
        ax3.set_ylabel('Densità')
        ax3.set_title('Distribuzione Probabilità', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. ROC Curve
        ax4 = plt.subplot(3, 4, 4)
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        ax4.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        ax4.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax4.set_xlim([0.0, 1.0])
        ax4.set_ylim([0.0, 1.05])
        ax4.set_xlabel('Tasso Falsi Positivi')
        ax4.set_ylabel('Tasso Veri Positivi')
        ax4.set_title('Curva ROC', fontsize=14, fontweight='bold')
        ax4.legend(loc="lower right")
        ax4.grid(True, alpha=0.3)

        # 5. Top 5 Features più importanti
        ax5 = plt.subplot(3, 4, 5)
        top_features = importance_df.nlargest(5, 'RF_Importance')
        bars = ax5.barh(range(len(top_features)), top_features['RF_Importance'], color='skyblue')
        ax5.set_yticks(range(len(top_features)))
        ax5.set_yticklabels(top_features['Feature'])
        ax5.set_xlabel('Importanza')
        ax5.set_title('Top 5 Features', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # Aggiungi valori sulle barre
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax5.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')

        # 6. Loss curve (se disponibile)
        ax6 = plt.subplot(3, 4, 6)
        if hasattr(self.model, 'loss_curve_'):
            ax6.plot(self.model.loss_curve_, color='red', linewidth=2)
            ax6.set_xlabel('Iterazioni')
            ax6.set_ylabel('Loss')
            ax6.set_title('Curva di Loss', fontsize=14, fontweight='bold')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'Loss curve\nnon disponibile', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Curva di Loss', fontsize=14, fontweight='bold')

        # 7. Distribuzione per classe
        ax7 = plt.subplot(3, 4, 7)
        class_dist = pd.Series(y_test).value_counts()
        wedges, texts, autotexts = ax7.pie(class_dist.values, labels=['Non Sopravv.', 'Sopravv.'], 
                                          autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
        ax7.set_title('Distribuzione Classi (Test Set)', fontsize=14, fontweight='bold')

        # 8. Calibration plot
        ax8 = plt.subplot(3, 4, 8)
        from sklearn.calibration import calibration_curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_pred_prob, n_bins=10)
        ax8.plot(mean_predicted_value, fraction_of_positives, "s-", 
                color='red', label='Modello')
        ax8.plot([0, 1], [0, 1], "k:", label="Perfettamente calibrato")
        ax8.set_xlabel('Probabilità media predetta')
        ax8.set_ylabel('Frazione di positivi')
        ax8.set_title('Calibration Plot', fontsize=14, fontweight='bold')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        # 9-12. Esempi di predizioni
        for i, ax_idx in enumerate([9, 10, 11, 12]):
            ax = plt.subplot(3, 4, ax_idx)
            ax.axis('off')

            # Prendi un esempio casuale
            idx = np.random.randint(0, len(y_test))
            actual = y_test.iloc[idx]
            predicted = y_pred[idx]
            prob = y_pred_prob[idx]

            result_color = 'green' if actual == predicted else 'red'
            result_text = '✅ CORRETTO' if actual == predicted else '❌ ERRATO'

            example_text = f"""
ESEMPIO {i+1}
━━━━━━━━━━━━━━━
🎯 Reale: {'Sopravv.' if actual else 'Non Sopravv.'}
🤖 Predetto: {'Sopravv.' if predicted else 'Non Sopravv.'}
📊 Probabilità: {prob:.3f}
━━━━━━━━━━━━━━━
{result_text}
            """

            ax.text(0.5, 0.5, example_text, fontsize=10, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=result_color, alpha=0.3))

        plt.suptitle('🚢 TITANIC NEURAL NETWORK - RIEPILOGO FINALE 🚢', 
                    fontsize=20, fontweight='bold', y=0.98)

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig('titanic_final_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Grafico riepilogo salvato come 'titanic_final_summary.png'")

# Esempio di utilizzo
def main():
    # Inizializza il modello
    titanic_nn = TitanicMLPNetwork()

    # Carica e preprocessa i dati
    df = titanic_nn.load_and_preprocess_data('dataset.csv')

    # Esplora i dati
    titanic_nn.explore_data(df)

    # Analizza l'importanza delle features
    X_all = df.drop('Survived', axis=1)
    y_all = df['Survived']
    importance_analysis = titanic_nn.analyze_feature_importance(X_all, y_all)
    print("\nTop 5 features più importanti:")
    print(importance_analysis.sort_values('RF_Importance', ascending=False)[['Feature', 'RF_Importance']].head())

    # Prepara le features
    X_train, X_test, y_train, y_test = titanic_nn.prepare_features(df)

    # Costruisce e addestra il modello
    model = titanic_nn.build_and_train_model(X_train, X_test, y_train, y_test)

    # Plot delle curve di apprendimento
    titanic_nn.plot_learning_curves(X_train, y_train)

    # Valuta il modello
    accuracy, predictions = titanic_nn.evaluate_model(X_test, y_test)

    # Crea il grafico di riepilogo finale
    titanic_nn.create_final_summary_plot(accuracy, y_test, predictions, importance_analysis)

    # Esempio di predizione per un nuovo passeggero
    new_passenger = {
        'Pclass': 1,           # Prima classe
        'Sex': 0,              # Femmina (dopo encoding)
        'Age': 25,             # 25 anni
        'SibSp': 0,            # Nessun fratello/sorella a bordo
        'Parch': 1,            # Un genitore/figlio a bordo
        'Fare': 80.0,          # Tariffa pagata
        'Embarked': 2,         # Imbarco (dopo encoding)
        'Has_Cabin': 1,        # Ha una cabina
        'Family_Size': 2,      # Dimensione famiglia
        'Is_Alone': 0,         # Non è sola
        'Title': 1,            # Titolo (dopo encoding)
        'Age_Group': 2,        # Gruppo età
        'Fare_Group': 3        # Gruppo tariffa
    }

    prob, prediction = titanic_nn.predict_passenger(new_passenger)

    print(f"\n🎉 Addestramento completato!")
    print(f"Accuracy finale: {accuracy:.4f}")

    # Informazioni aggiuntive sul modello
    print(f"\n📋 Informazioni sul modello:")
    print(f"- Numero di layer: {len(titanic_nn.model.hidden_layer_sizes) + 1}")
    print(f"- Neuroni per layer: {titanic_nn.model.hidden_layer_sizes}")
    print(f"- Numero totale di parametri: {sum([layer.size for layer in titanic_nn.model.coefs_]) + sum([layer.size for layer in titanic_nn.model.intercepts_])}")

if __name__ == "__main__":
    main()

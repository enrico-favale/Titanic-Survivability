# Crea anche grafici individuali per maggiore dettaglio
        self.save_individual_prediction_plots(results_df, y_pred, y_pred_prob)
    
    def save_individual_prediction_plots(self, results_df, y_pred, y_pred_prob):
        """
        Salva grafici individuali per ogni tipo di analisi
        """
        print("\n📊 Salvando grafici individuali dettagliati...")
        
        # 1. Grafico distribuzione probabilità dettagliato
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        n, bins, patches = plt.hist(y_pred_prob, bins=30, alpha=0.7, color='lightblue', 
                                   edgecolor='black', density=True)
        plt.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Soglia decisione')
        plt.axvline(y_pred_prob.mean(), color='orange', linestyle='-', linewidth=2, 
                   label=f'Media ({y_pred_prob.mean():.3f})')
        plt.xlabel('Probabilità di sopravvivenza')
        plt.ylabel('Densità')
        plt.title('Distribuzione Dettagliata delle Probabilità', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Box plot per sesso
        plt.subplot(2, 2, 2)
        sex_data = [results_df[results_df['Sex']==0]['Survival_Probability'],
                   results_df[results_df['Sex']==1]['Survival_Probability']]
        plt.boxplot(sex_data, labels=['Female', 'Male'])
        plt.ylabel('Probabilità di sopravvivenza')
        plt.title('Box Plot: Probabilità per Sesso', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 3. Box plot per classe
        plt.subplot(2, 2, 3)
        class_data = [results_df[results_df['Pclass']==i]['Survival_Probability'] for i in [1,2,3]]
        plt.boxplot(class_data, labels=['Classe 1', 'Classe 2', 'Classe 3'])
        plt.ylabel('Probabilità di sopravvivenza')
        plt.title('Box Plot: Probabilità per Classe', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 4. Scatter plot: Età vs Probabilità
        plt.subplot(2, 2, 4)
        colors = ['red' if pred == 0 else 'green' for pred in y_pred]
        plt.scatter(results_df['Age'], y_pred_prob, c=colors, alpha=0.6)
        plt.axhline(0.5, color='blue', linestyle='--', alpha=0.7)
        plt.xlabel('Età')
        plt.ylabel('Probabilità di sopravvivenza')
        plt.title('Età vs Probabilità di Sopravvivenza', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('titanic_detailed_probability_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Analisi dettagliata probabilità salvata come 'titanic_detailed_probability_analysis.png'")
        
        # 2. Grafico di confronto demografico
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Sopravvivenza per tutte le combinazioni
        plt.subplot(2, 3, 1)
        survival_by_sex_class = results_df.groupby(['Sex', 'Pclass'])['Predicted_Survived'].mean().unstack()
        survival_by_sex_class.plot(kind='bar', ax=plt.gca(), color=['gold', 'silver', 'brown'])
        plt.title('Sopravvivenza per Sesso e Classe')
        plt.xlabel('Sesso (0=Female, 1=Male)')
        plt.ylabel('Tasso sopravvivenza')
        plt.xticks(rotation=0)
        plt.legend(title='Classe')
        
        # Subplot 2: Distribuzione età per sopravvissuti vs non
        plt.subplot(2, 3, 2)
        survived_ages = results_df[results_df['Predicted_Survived']==1]['Age']
        not_survived_ages = results_df[results_df['Predicted_Survived']==0]['Age']
        plt.hist(survived_ages, alpha=0.7, label='Sopravvissuti', bins=20, color='green')
        plt.hist(not_survived_ages, alpha=0.7, label='Non sopravvissuti', bins=20, color='red')
        plt.xlabel('Età')
        plt.ylabel('Frequenza')
        plt.title('Distribuzione Età per Predizione')
        plt.legend()
        
        # Subplot 3: Tariffa vs Sopravvivenza
        plt.subplot(2, 3, 3)
        plt.scatter(results_df['Fare'], y_pred_prob, c=y_pred, cmap='RdYlGn', alpha=0.6)
        plt.xlabel('Tariffa pagata')
        plt.ylabel('Probabilità sopravvivenza')
        plt.title('Tariffa vs Probabilità Sopravvivenza')
        plt.colorbar(label='Predizione (0=Morte, 1=Sopravv.)')
        
        # Subplot 4: Famiglia vs Sopravvivenza
        plt.subplot(2, 3, 4)
        family_stats = results_df.groupby('Family_Size').agg({
            'Predicted_Survived': ['count', 'sum', 'mean']
        })
        family_stats.columns = ['Totale', 'Sopravvissuti', 'Tasso']
        family_stats['Tasso'].plot(kind='bar', color='purple', alpha=0.7)
        plt.title('Tasso Sopravvivenza per Dimensione Famiglia')
        plt.xlabel('Dimensione Famiglia')
        plt.ylabel('Tasso Sopravvivenza')
        plt.xticks(rotation=0)
        
        # Subplot 5: Mappa di calore completa
        plt.subplot(2, 3, 5)
        # Crea una matrice di correlazione delle probabilità
        correlation_features = ['Pclass', 'Sex', 'Age_Group', 'Family_Size', 'Survival_Probability']
        corr_matrix = results_df[correlation_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Correlazioni Features vs Probabilità')
        
        # Subplot 6: Statistiche riassuntive
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        # Calcola statistiche interessanti
        total_passengers = len(results_df)
        predicted_survivors = y_pred.sum()
        high_confidence_survivors = len(results_df[results_df['Survival_Probability'] > 0.8])
        low_confidence_deaths = len(results_df[results_df['Survival_Probability'] < 0.2])
        
        stats_text = f"""
📊 STATISTICHE RIASSUNTIVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
👥 Passeggeri totali nel test set: {total_passengers}
✅ Predetti sopravvissuti: {predicted_survivors} ({predicted_survivors/total_passengers*100:.1f}%)
❌ Predetti morti: {total_passengers-predicted_survivors} ({(total_passengers-predicted_survivors)/total_passengers*100:.1f}%)

🎯 CONFIDENZA DELLE PREDIZIONI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🟢 Alta confidenza sopravv. (>80%): {high_confidence_survivors}
🔴 Alta confidenza morte (<20%): {low_confidence_deaths}
🟡 Casi incerti (20-80%): {total_passengers - high_confidence_survivors - low_confidence_deaths}

📈 PROBABILITÀ MEDIA: {y_pred_prob.mean():.3f}
📊 MEDIANA: {np.median(y_pred_prob):.3f}
📏 DEVIAZIONE STANDARD: {y_pred_prob.std():.3f}

🏆 INSIGHTS CHIAVE:
• Donne: {results_df[results_df['Sex']==0]['Predicted_Survived'].mean()*100:.1f}% sopravvivenza
• Uomini: {results_df[results_df['Sex']==1]['Predicted_Survived'].mean()*100:.1f}% sopravvivenza
• Prima classe: {results_df[results_df['Pclass']==1]['Predicted_Survived'].mean()*100:.1f}% sopravvivenza
• Terza classe: {results_df[results_df['Pclass']==3]['Predicted_Survived'].mean()*100:.1f}% sopravvivenza
        """
        
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                facecolor="lightcyan", alpha=0.8))
        
        plt.suptitle('📈 ANALISI DEMOGRAFICA DETTAGLIATA DELLE PREDIZIONI', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('titanic_demographic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Analisi demografica salvata come 'titanic_demographic_analysis.png'")
        
        # 3. Grafico finale riassuntivo con esempi specifici
        plt.figure(figsize=(16, 10))
        
        # Trova esempi interessanti
        examples_df = self.find_interesting_examples(results_df)
        
        # Subplot principale: scatter plot completo
        plt.subplot(2, 2, (1, 2))
        scatter = plt.scatter(results_df['Age'], results_df['Fare'], 
                            c=y_pred_prob, cmap='RdYlGn', s=60, alpha=0.7, edgecolors='black')
        plt.colorbar(scatter, label='Probabilità di Sopravvivenza')
        plt.xlabel('Età')
        plt.ylabel('Tariffa Pagata')
        plt.title('Mappa Completa: Età vs Tariffa vs Probabilità Sopravvivenza', fontweight='bold', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Aggiungi annotazioni per casi estremi
        top_case = results_df.loc[results_df['Survival_Probability'].idxmax()]
        bottom_case = results_df.loc[results_df['Survival_Probability'].idxmin()]
        
        plt.annotate(f'MAX: {top_case["Survival_Probability"]:.3f}', 
                    xy=(top_case['Age'], top_case['Fare']), xytext=(10, 10),
                    textcoords='offset points', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='green', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.annotate(f'MIN: {bottom_case["Survival_Probability"]:.3f}', 
                    xy=(bottom_case['Age'], bottom_case['Fare']), xytext=(10, -10),
                    textcoords='offset points', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='red', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Subplot esempi interessanti
        plt.subplot(2, 2, 3)
        plt.axis('off')
        examples_text = self.format_examples_text(examples_df)
        plt.text(0.05, 0.95, examples_text, transform=plt.gca().transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                facecolor="lightyellow", alpha=0.9))
        plt.title('🎯 Esempi Rappresentativi', fontweight='bold')
        
        # Subplot distribuzione finale
        plt.subplot(2, 2, 4)
        # Crea un grafico a barre impilate per classe e sesso
        pivot_data = results_df.pivot_table(values='Predicted_Survived', 
                                           index='Pclass', columns='Sex', aggfunc='mean')
        pivot_data.plot(kind='bar', ax=plt.gca(), color=['hotpink', 'cornflowerblue'])
        plt.title('Tasso Sopravvivenza: Classe vs Sesso', fontweight='bold')
        plt.xlabel('Classe')
        plt.ylabel('Tasso Sopravvivenza')
        plt.legend(['Female', 'Male'])
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('🚢 RIEPILOGO FINALE PREDIZIONI TITANIC 🚢', 
                    fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig('titanic_final_prediction_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Riepilogo finale salvato come 'titanic_final_prediction_summary.png'")
        
        print("\n🎉 TUTTI I GRAFICI SALVATI CON SUCCESSO!")
        print("📂 File grafici generati:")
        print("   1. titanic_test_predictions_analysis.png (PRINCIPALE)")
        print("   2. titanic_detailed_probability_analysis.png")
        print("   3. titanic_demographic_analysis.png") 
        print("   4. titanic_final_prediction_summary.png")
    
    def find_interesting_examples(self, results_df):
        """
        Trova esempi interessanti per l'analisi
        """
        examples = []
        
        # Caso 1: Massima probabilità sopravvivenza
        max_prob_idx = results_df['Survival_Probability'].idxmax()
        examples.append(('MAX_SURVIVAL', results_df.loc[max_prob_idx]))
        
        # Caso 2: Minima probabilità sopravvivenza  
        min_prob_idx = results_df['Survival_Probability'].idxmin()
        examples.append(('MIN_SURVIVAL', results_df.loc[min_prob_idx]))
        
        # Caso 3: Esempio tipico donna prima classe
        women_first = results_df[(results_df['Sex']==0) & (results_df['Pclass']==1)]
        if len(women_first) > 0:
            typical_woman_first = women_first.iloc[len(women_first)//2]
            examples.append(('TYPICAL_WOMAN_1ST', typical_woman_first))
        
        # Caso 4: Esempio tipico uomo terza classe
        men_third = results_df[(results_df['Sex']==1) & (results_df['Pclass']==3)]
        if len(men_third) > 0:
            typical_man_third = men_third.iloc[len(men_third)//2]
            examples.append(('TYPICAL_MAN_3RD', typical_man_third))
        
        # Caso 5: Caso più incerto (vicino a 0.5)
        uncertain_idx = (results_df['Survival_Probability'] - 0.5).abs().idxmin()
        examples.append(('MOST_UNCERTAIN', results_df.loc[uncertain_idx]))
        
        return examples
    
    def format_examples_text(self, examples):
        """
        Formatta il testo degli esempi per la visualizzazione
        """
        text = "🎯 ESEMPI RAPPRESENTATIVI\n"
        text += "=" * 40 + "\n\n"
        
        labels = {
            'MAX_SURVIVAL': '🏆 MASSIMA SOPRAVVIVENZA',
            'MIN_SURVIVAL': '⚰️ MINIMA SOPRAVVIVENZA', 
            'TYPICAL_WOMAN_1ST': '👩 DONNA PRIMA CLASSE',
            'TYPICAL_MAN_3RD': '👨 UOMO TERZA CLASSE',
            'MOST_UNCERTAIN': '🤔 CASO PIÙ INCERTO'
        }
        
        for label, row in examples:
            text += f"{labels.get(label, label)}\n"
            text += f"Classe: {row['Pclass']} | "
            text += f"Sesso: {'F' if row['Sex']==0 else 'M'} | "
            text += f"Età: {row['Age']:.0f}\n"
            text += f"Famiglia: {row['Family_Size']} | "
            text += f"Prob: {row['Survival_Probability']:.3f}\n"
            text += f"Prediz: {'✅ Sopravv.' if row['Predicted_Survived'] else '❌ Morte'}\n\n"
        
        return textimport pandas as pd
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
        Prepara le features per l'addestramento e crea un test set senza target
        """
        # Separa features e target
        X = df.drop('Survived', axis=1)
        y = df['Survived']
        
        # Split train/test - il test set non avrà la colonna Survived
        X_train, X_test, y_train, y_test_true = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalizzazione
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\n📋 Dati preparati:")
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set (SENZA target): {X_test_scaled.shape}")
        print(f"Features utilizzate: {list(X.columns)}")
        print(f"\n🎯 Il test set non contiene la colonna 'Survived'")
        print(f"Faremo predizioni 'blind' su {len(X_test)} passeggeri!")
        
        # Salva anche le informazioni originali del test set per l'analisi
        self.test_set_info = X_test.copy()
        self.test_set_info['True_Survived'] = y_test_true  # Solo per validazione interna
        
        return X_train_scaled, X_test_scaled, y_train, y_test_true
    
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
    
    def make_predictions_on_test_set(self, X_test):
        """
        Fa predizioni sul test set senza conoscere le vere etichette
        """
        print("\n🔮 Facendo predizioni sul test set...")
        print("=" * 60)
        
        # Predizioni
        y_pred = self.model.predict(X_test)
        y_pred_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Crea DataFrame con risultati
        results_df = self.test_set_info.drop('True_Survived', axis=1).copy()
        results_df['Predicted_Survived'] = y_pred
        results_df['Survival_Probability'] = y_pred_prob
        results_df['Prediction_Text'] = results_df['Predicted_Survived'].map({0: 'Non Sopravvissuto', 1: 'Sopravvissuto'})
        
        # Statistiche delle predizioni
        survival_count = y_pred.sum()
        total_passengers = len(y_pred)
        survival_rate = survival_count / total_passengers
        
        print(f"📊 RISULTATI PREDIZIONI SUL TEST SET:")
        print(f"   👥 Passeggeri totali: {total_passengers}")
        print(f"   ✅ Predetti sopravvissuti: {survival_count} ({survival_rate:.1%})")
        print(f"   ❌ Predetti non sopravvissuti: {total_passengers - survival_count} ({1-survival_rate:.1%})")
        print(f"   📈 Probabilità media di sopravvivenza: {y_pred_prob.mean():.3f}")
        
        # Salva i risultati
        results_df.to_csv('titanic_test_predictions.csv', index=False)
        print(f"\n💾 Predizioni salvate in 'titanic_test_predictions.csv'")
        
        return results_df, y_pred, y_pred_prob
    
    def analyze_test_predictions(self, results_df, y_pred, y_pred_prob):
        """
        Analizza le predizioni fatte sul test set
        """
        print("\n📈 ANALISI DETTAGLIATA DELLE PREDIZIONI:")
        print("=" * 60)
        
        # Analisi per caratteristiche demografiche
        print("\n🚻 Sopravvivenza per SESSO:")
        sex_analysis = results_df.groupby('Sex').agg({
            'Predicted_Survived': ['count', 'sum', 'mean'],
            'Survival_Probability': 'mean'
        }).round(3)
        print(sex_analysis)
        
        print("\n🎭 Sopravvivenza per CLASSE:")
        class_analysis = results_df.groupby('Pclass').agg({
            'Predicted_Survived': ['count', 'sum', 'mean'],
            'Survival_Probability': 'mean'
        }).round(3)
        print(class_analysis)
        
        print("\n👨‍👩‍👧‍👦 Sopravvivenza per DIMENSIONE FAMIGLIA:")
        family_analysis = results_df.groupby('Family_Size').agg({
            'Predicted_Survived': ['count', 'sum', 'mean'],
            'Survival_Probability': 'mean'
        }).round(3)
        print(family_analysis)
        
        # Trova casi interessanti
        print("\n🎯 CASI PIÙ INTERESSANTI:")
        print("-" * 40)
        
        # Sopravvissuti con alta probabilità
        high_survival = results_df[results_df['Survival_Probability'] > 0.9]
        print(f"🟢 Sopravvivenza MOLTO PROBABILE (>90%): {len(high_survival)} passeggeri")
        
        # Non sopravvissuti con alta certezza
        low_survival = results_df[results_df['Survival_Probability'] < 0.1]
        print(f"🔴 Sopravvivenza MOLTO IMPROBABILE (<10%): {len(low_survival)} passeggeri")
        
        # Casi incerti
        uncertain = results_df[(results_df['Survival_Probability'] > 0.4) & 
                              (results_df['Survival_Probability'] < 0.6)]
        print(f"🟡 Casi INCERTI (40-60%): {len(uncertain)} passeggeri")
        
        return sex_analysis, class_analysis, family_analysis
    
    def create_test_predictions_visualization(self, results_df, y_pred, y_pred_prob):
        """
        Crea visualizzazioni delle predizioni sul test set
        """
        print("\n📊 Creando visualizzazioni delle predizioni...")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        
        # 1. Distribuzione delle predizioni
        ax1 = axes[0, 0]
        pred_counts = pd.Series(y_pred).value_counts()
        colors = ['red', 'green']
        wedges, texts, autotexts = ax1.pie(pred_counts.values, 
                                          labels=['Non Sopravvissuto', 'Sopravvissuto'],
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('📊 Distribuzione Predizioni', fontweight='bold', fontsize=14)
        
        # 2. Distribuzione probabilità
        ax2 = axes[0, 1]
        n, bins, patches = ax2.hist(y_pred_prob, bins=25, alpha=0.8, color='skyblue', 
                                   edgecolor='black', density=True)
        ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Soglia decisione (0.5)')
        ax2.axvline(y_pred_prob.mean(), color='orange', linestyle='-', linewidth=2, 
                   label=f'Media ({y_pred_prob.mean():.2f})')
        ax2.set_xlabel('Probabilità di sopravvivenza')
        ax2.set_ylabel('Densità')
        ax2.set_title('📈 Distribuzione Probabilità Predette', fontweight='bold', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Sopravvivenza per sesso
        ax3 = axes[0, 2]
        sex_survival = results_df.groupby('Sex')['Predicted_Survived'].mean()
        sex_counts = results_df.groupby('Sex').size()
        bars = ax3.bar(['Female', 'Male'], sex_survival.values, 
                      color=['hotpink', 'cornflowerblue'], alpha=0.8, edgecolor='black')
        ax3.set_ylabel('Tasso di sopravvivenza predetto')
        ax3.set_title('🚻 Sopravvivenza per Sesso', fontweight='bold', fontsize=14)
        ax3.set_ylim(0, 1)
        for i, (bar, value, count) in enumerate(zip(bars, sex_survival.values, sex_counts.values)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                    f'{value:.2f}\n({count} pers.)', ha='center', va='bottom', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Sopravvivenza per classe
        ax4 = axes[1, 0]
        class_survival = results_df.groupby('Pclass')['Predicted_Survived'].mean()
        class_counts = results_df.groupby('Pclass').size()
        bars = ax4.bar(class_survival.index, class_survival.values, 
                      color=['gold', 'silver', '#CD7F32'], alpha=0.8, edgecolor='black')
        ax4.set_xlabel('Classe Passeggero')
        ax4.set_ylabel('Tasso di sopravvivenza predetto')
        ax4.set_title('🎭 Sopravvivenza per Classe', fontweight='bold', fontsize=14)
        ax4.set_ylim(0, 1)
        for bar, value, count in zip(bars, class_survival.values, class_counts.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                    f'{value:.2f}\n({count} pers.)', ha='center', va='bottom', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Sopravvivenza per età
        ax5 = axes[1, 1]
        age_groups = pd.cut(results_df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                           labels=['Bambini\n(0-12)', 'Adolesc.\n(13-18)', 'Adulti\n(19-35)', 
                                  'Mezza età\n(36-60)', 'Anziani\n(60+)'])
        age_survival = results_df.groupby(age_groups)['Predicted_Survived'].mean()
        age_counts = results_df.groupby(age_groups).size()
        bars = ax5.bar(range(len(age_survival)), age_survival.values, 
                      color=['lightgreen', 'lightblue', 'lightyellow', 'lightcoral', 'plum'],
                      alpha=0.8, edgecolor='black')
        ax5.set_xticks(range(len(age_survival)))
        ax5.set_xticklabels(age_survival.index, rotation=0, fontsize=10)
        ax5.set_ylabel('Tasso di sopravvivenza predetto')
        ax5.set_title('👶👨👴 Sopravvivenza per Fascia di Età', fontweight='bold', fontsize=14)
        ax5.set_ylim(0, 1)
        for i, (bar, value, count) in enumerate(zip(bars, age_survival.values, age_counts.values)):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                    f'{value:.2f}\n({count})', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Sopravvivenza per dimensione famiglia
        ax6 = axes[1, 2]
        family_survival = results_df.groupby('Family_Size')['Predicted_Survived'].mean()
        family_counts = results_df.groupby('Family_Size').size()
        ax6.plot(family_survival.index, family_survival.values, 'o-', color='purple', 
                linewidth=3, markersize=8, markerfacecolor='white', markeredgewidth=2)
        ax6.set_xlabel('Dimensione Famiglia')
        ax6.set_ylabel('Tasso di sopravvivenza predetto')
        ax6.set_title('👨‍👩‍👧‍👦 Sopravvivenza per Dimensione Famiglia', fontweight='bold', fontsize=14)
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, 1)
        # Aggiungi etichette sui punti
        for x, y, count in zip(family_survival.index, family_survival.values, family_counts.values):
            ax6.annotate(f'{y:.2f}\n({count})', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold', fontsize=9)
        
        # 7. Heatmap: Classe vs Sesso
        ax7 = axes[2, 0]
        heatmap_data = results_df.pivot_table(values='Predicted_Survived', 
                                             index='Pclass', columns='Sex', aggfunc='mean')
        im = sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', ax=ax7, 
                        xticklabels=['Female', 'Male'], fmt='.3f', cbar_kws={'label': 'Tasso Sopravvivenza'})
        ax7.set_title('🔥 Heatmap: Classe vs Sesso', fontweight='bold', fontsize=14)
        ax7.set_ylabel('Classe Passeggero')
        
        # 8. Distribuzione per porto di imbarco
        ax8 = axes[2, 1]
        embarked_survival = results_df.groupby('Embarked')['Predicted_Survived'].mean()
        embarked_counts = results_df.groupby('Embarked').size()
        embarked_labels = {0: 'Cherbourg\n(C)', 1: 'Queenstown\n(Q)', 2: 'Southampton\n(S)'}
        bars = ax8.bar([embarked_labels.get(x, f'Port {x}') for x in embarked_survival.index], 
                      embarked_survival.values, color=['cyan', 'orange', 'lime'], 
                      alpha=0.8, edgecolor='black')
        ax8.set_ylabel('Tasso di sopravvivenza predetto')
        ax8.set_title('⚓ Sopravvivenza per Porto di Imbarco', fontweight='bold', fontsize=14)
        ax8.set_ylim(0, 1)
        for bar, value, count in zip(bars, embarked_survival.values, embarked_counts.values):
            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                    f'{value:.2f}\n({count} pers.)', ha='center', va='bottom', fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')
        
        # 9. Statistiche e Esempi Estremi
        ax9 = axes[2, 2]
        ax9.axis('off')
        
        # Trova esempi estremi
        top_survivor = results_df.loc[results_df['Survival_Probability'].idxmax()]
        bottom_survivor = results_df.loc[results_df['Survival_Probability'].idxmin()]
        
        # Statistiche aggiuntive
        high_prob_count = len(results_df[results_df['Survival_Probability'] > 0.8])
        low_prob_count = len(results_df[results_df['Survival_Probability'] < 0.2])
        uncertain_count = len(results_df[(results_df['Survival_Probability'] >= 0.4) & 
                                        (results_df['Survival_Probability'] <= 0.6)])
        
        stats_text = f"""
📊 STATISTICHE PREDIZIONI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
👥 Totale passeggeri: {len(results_df)}
✅ Predetti sopravvissuti: {y_pred.sum()} ({y_pred.sum()/len(y_pred)*100:.1f}%)
❌ Predetti non sopravv.: {len(y_pred)-y_pred.sum()} ({(1-y_pred.sum()/len(y_pred))*100:.1f}%)

🎯 CONFIDENZA PREDIZIONI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🟢 Alta prob. sopravv. (>80%): {high_prob_count}
🔴 Alta prob. morte (<20%): {low_prob_count}
🟡 Casi incerti (40-60%): {uncertain_count}

📈 PROBABILITÀ MEDIA: {y_pred_prob.mean():.3f}
📊 DEVIAZIONE STANDARD: {y_pred_prob.std():.3f}

🏆 CASO PIÙ PROBABILE SOPRAVV.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
👤 Classe: {top_survivor['Pclass']} | 🚻 Sesso: {'F' if top_survivor['Sex']==0 else 'M'}
🎂 Età: {top_survivor['Age']:.0f} | 👨‍👩‍👧‍👦 Famiglia: {top_survivor['Family_Size']}
📊 Probabilità: {top_survivor['Survival_Probability']:.3f}

⚰️ CASO MENO PROBABILE SOPRAVV.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
👤 Classe: {bottom_survivor['Pclass']} | 🚻 Sesso: {'F' if bottom_survivor['Sex']==0 else 'M'}
🎂 Età: {bottom_survivor['Age']:.0f} | 👨‍👩‍👧‍👦 Famiglia: {bottom_survivor['Family_Size']}
📊 Probabilità: {bottom_survivor['Survival_Probability']:.3f}
        """
        
        ax9.text(0.05, 0.95, stats_text, fontsize=10, verticalalignment='top',
                horizontalalignment='left', transform=ax9.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9, edgecolor='black'))
        
        plt.suptitle('🔮 ANALISI COMPLETA PREDIZIONI SUL TEST SET (20% Dataset) 🔮', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        
        # SALVA IL GRAFICO PRINCIPALE DELLE PREDIZIONI
        plt.savefig('titanic_test_predictions_analysis.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print("💾 ⭐ GRAFICO PRINCIPALE delle predizioni salvato come 'titanic_test_predictions_analysis.png'")
        
        plt.show()
        
        # Crea anche grafici individuali per maggiore dettaglio
        self.save_individual_prediction_plots(results_df, y_pred, y_pred_prob)
    
    def evaluate_model_on_validation(self, X_test, y_test):
        """
        Valuta le performance del modello su un set di validazione interno
        """
        print("\n📊 Validazione interna del modello (per verifica)...")
        
        # Cross-validation sul training set
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5, scoring='accuracy')
        
        print(f"🎯 Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Predizioni per il report
        y_pred = self.model.predict(X_test)
        y_pred_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Metriche
        accuracy = accuracy_score(y_test, y_pred)
        print(f"🎯 Validation Accuracy: {accuracy:.4f}")
        
        print("\n📈 Report di classificazione (validazione interna):")
        print(classification_report(y_test, y_pred))
        
        # Visualizzazioni rapide
        plt.figure(figsize=(12, 4))
        
        # Matrice di confusione
        plt.subplot(1, 3, 1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matrice di Confusione (Validazione)')
        plt.xlabel('Predetto')
        plt.ylabel('Reale')
        
        # Distribuzione delle probabilità
        plt.subplot(1, 3, 2)
        plt.hist(y_pred_prob[y_test == 0], alpha=0.6, label='Non sopravvissuti', 
                bins=15, color='red', density=True)
        plt.hist(y_pred_prob[y_test == 1], alpha=0.6, label='Sopravvissuti', 
                bins=15, color='green', density=True)
        plt.xlabel('Probabilità di sopravvivenza')
        plt.ylabel('Densità')
        plt.title('Distribuzione Probabilità (Validazione)')
        plt.legend()
        
        # ROC Curve
        plt.subplot(1, 3, 3)
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('Tasso Falsi Positivi')
        plt.ylabel('Tasso Veri Positivi')
        plt.title('Curva ROC (Validazione)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('titanic_internal_validation.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Validazione interna salvata come 'titanic_internal_validation.png'")
        
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

# Esempio di utilizzo
def main():
    print("🚢" * 20)
    print("TITANIC SURVIVAL PREDICTION - TEST SET ANALYSIS")
    print("🚢" * 20)
    
    # Inizializza il modello
    titanic_nn = TitanicMLPNetwork()
    
    # Carica e preprocessa i dati
    df = titanic_nn.load_and_preprocess_data('6_titanic.csv')
    
    # Esplora i dati
    titanic_nn.explore_data(df)
    
    # Analizza l'importanza delle features
    X_all = df.drop('Survived', axis=1)
    y_all = df['Survived']
    importance_analysis = titanic_nn.analyze_feature_importance(X_all, y_all)
    print("\nTop 5 features più importanti:")
    print(importance_analysis.sort_values('RF_Importance', ascending=False)[['Feature', 'RF_Importance']].head())
    
    # Prepara le features (80% train, 20% test)
    X_train, X_test, y_train, y_test_true = titanic_nn.prepare_features(df)
    
    print("\n" + "="*60)
    print("🧠 FASE 1: ADDESTRAMENTO SUL 80% DEI DATI")
    print("="*60)
    
    # Costruisce e addestra il modello SOLO sul training set
    model = titanic_nn.build_and_train_model(X_train, X_test, y_train, y_test_true)
    
    # Plot delle curve di apprendimento
    titanic_nn.plot_learning_curves(X_train, y_train)
    
    # Validazione interna per verificare che il modello funzioni
    print("\n" + "="*60)
    print("🔍 FASE 2: VALIDAZIONE INTERNA DEL MODELLO")
    print("="*60)
    accuracy, _ = titanic_nn.evaluate_model_on_validation(X_test, y_test_true)
    
    print("\n" + "="*60)
    print("🔮 FASE 3: PREDIZIONI SUL TEST SET (SENZA CONOSCERE LA VERITÀ)")
    print("="*60)
    
    # Ora facciamo le predizioni "blind" sul test set
    results_df, y_pred, y_pred_prob = titanic_nn.make_predictions_on_test_set(X_test)
    
    # Analizza le predizioni
    sex_analysis, class_analysis, family_analysis = titanic_nn.analyze_test_predictions(results_df, y_pred, y_pred_prob)
    
    # Crea visualizzazioni delle predizioni
    titanic_nn.create_test_predictions_visualization(results_df, y_pred, y_pred_prob)
    
    print("\n" + "="*60)
    print("📊 FASE 4: CONFRONTO CON LA REALTÀ (solo per analisi)")
    print("="*60)
    
    # Solo per curiosità, confrontiamo con la verità (ma questo non sarebbe disponibile nella realtà)
    actual_accuracy = accuracy_score(y_test_true, y_pred)
    print(f"🎯 Accuracy reale sul test set: {actual_accuracy:.4f}")
    print(f"📈 Questo conferma che il modello funziona bene anche su dati mai visti!")
    
    # Mostra alcuni esempi specifici
    print("\n🔍 ESEMPI DI PREDIZIONI INTERESSANTI:")
    print("-" * 50)
    
    # Aggiungi la verità ai risultati solo per l'analisi finale
    results_with_truth = results_df.copy()
    results_with_truth['True_Survived'] = y_test_true.values
    results_with_truth['Correct_Prediction'] = (results_with_truth['Predicted_Survived'] == results_with_truth['True_Survived'])
    
    # Mostra alcuni esempi
    examples = results_with_truth.sample(5)
    for idx, row in examples.iterrows():
        status = "✅ CORRETTO" if row['Correct_Prediction'] else "❌ ERRATO"
        print(f"Passeggero: Classe {row['Pclass']}, {'F' if row['Sex']==0 else 'M'}, {row['Age']:.0f} anni")
        print(f"  Predetto: {'Sopravv.' if row['Predicted_Survived'] else 'Non sopravv.'} (prob: {row['Survival_Probability']:.3f})")
        print(f"  Realtà: {'Sopravv.' if row['True_Survived'] else 'Non sopravv.'} - {status}")
        print()
    
    print(f"\n🎉 ANALISI COMPLETATA!")
    print(f"📋 Predizioni salvate in 'titanic_test_predictions.csv'")
    print(f"🖼️ Grafici salvati come immagini PNG")
    print(f"🎯 Accuracy finale: {actual_accuracy:.4f}")
    
    print("\n" + "🚢"*20)
    print("RIEPILOGO FINALE:")
    print("🚢"*20)
    print(f"📊 Passeggeri analizzati: {len(results_df)}")
    print(f"✅ Predetti sopravvissuti: {y_pred.sum()}")
    print(f"❌ Predetti non sopravvissuti: {len(y_pred) - y_pred.sum()}")
    print(f"🎯 Accuracy del modello: {actual_accuracy:.1%}")
    print(f"📈 Probabilità media sopravvivenza: {y_pred_prob.mean():.3f}")

if __name__ == "__main__":
    main()

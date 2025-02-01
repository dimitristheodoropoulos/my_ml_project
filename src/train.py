import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  # Εισαγωγή SMOTE
from scipy.stats import randint

# Φόρτωση δεδομένων
df = pd.read_csv('data/diabetes.csv')

# Χαρακτηριστικά και ετικέτες
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Κανονικοποίηση δεδομένων
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Εφαρμογή SMOTE για oversampling
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_scaled, y)

# Διαχωρισμός δεδομένων (με την επαναδειγματοληπτημένη εκδοχή)
X_train, X_test, y_train, y_test = train_test_split(X_train_resampled, y_train_resampled, test_size=0.2, random_state=42)

# Ορισμός υπερπαραμέτρων για GridSearchCV
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', None]
}

param_grid_xgb = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Λίστα μοντέλων
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "SVC": SVC(kernel='rbf', probability=True, class_weight="balanced", random_state=42),
    "LogisticRegression": LogisticRegression(class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False),  # Χωρίς use_label_encoder=True
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
}

# Βελτιστοποίηση υπερπαραμέτρων με GridSearchCV
best_model = None
best_score = 0

for name, model in models.items():
    try:
        # Χρήση GridSearchCV για RandomForest
        if name == "RandomForest":
            grid_search = GridSearchCV(model, param_grid_rf, cv=5, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model_rf = grid_search.best_estimator_
            model = best_model_rf
            print(f"\nΚαλύτερες υπερπαράμετροι για Random Forest: {grid_search.best_params_}")
        
        # Χρήση GridSearchCV για XGBoost
        elif name == "XGBoost":
            grid_search = GridSearchCV(model, param_grid_xgb, cv=5, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model_xgb = grid_search.best_estimator_
            model = best_model_xgb
            print(f"\nΚαλύτερες υπερπαράμετροι για XGBoost: {grid_search.best_params_}")

        # Εκπαίδευση του μοντέλου
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\n📌 Model: {name}")
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        if acc > best_score:
            best_score = acc
            best_model = model
    except Exception as e:
        print(f"Σφάλμα με το μοντέλο {name}: {e}")

# Αποθήκευση του καλύτερου μοντέλου
joblib.dump(best_model, 'models/diabetes_model.pkl')
print("\n✅ Το καλύτερο μοντέλο αποθηκεύτηκε!")

# Diabetes Prediction Project

## 📊 Περιγραφή

Αυτό το project αφορά την ανάπτυξη ενός συστήματος πρόβλεψης διαβήτη, βασισμένου σε τεχνικές μηχανικής μάθησης. Χρησιμοποιούμε διάφορους αλγορίθμους ταξινόμησης για την ανάλυση δεδομένων από το dataset διαβήτη του Kaggle και βελτιστοποιούμε τα αποτελέσματα μέσω υπερπαραμετροποίησης και τεχνικών επεξεργασίας δεδομένων.

---

## ⚙️ Χαρακτηριστικά

- **Προεπεξεργασία Δεδομένων:** Κανονικοποίηση με `StandardScaler`.
- **Αντιμετώπιση Ανομοιόμορφων Δεδομένων:** Χρήση `SMOTE` για oversampling.
- **Εκπαίδευση Μοντέλων:** Random Forest, SVM, Logistic Regression, XGBoost, AdaBoost, και Neural Networks.
- **Βελτιστοποίηση Υπερπαραμέτρων:** Grid Search για Random Forest και XGBoost.
- **Αξιολόγηση:** Classification report, confusion matrix και ακρίβεια.
- **Αυτόματη Αποθήκευση Καλύτερου Μοντέλου:** Χρήση `joblib`.

---

## 🗂️ Δομή Αρχείων

```
my_ml_project/
├── models/
│   └── diabetes_model.pkl  # Αποθηκευμένο εκπαιδευμένο μοντέλο
├── src/
│   ├── train.py            # Εκπαίδευση μοντέλων
│   └── predict.py          # Πρόβλεψη για νέα δεδομένα
├── .gitignore              # Αρχεία προς εξαίρεση από το Git
├── README.md               # Τεκμηρίωση του project
└── requirements.txt        # Εξαρτήσεις του project
```

---

## 🚀 Οδηγίες Εγκατάστασης

1. **Κλωνοποίηση του αποθετηρίου:**
   ```bash
   git clone https://github.com/dimitristheodoropoulos/diabetes-prediction.git
   cd diabetes-prediction
   ```

2. **Δημιουργία virtual περιβάλλοντος:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Για Linux
   venv\Scripts\activate     # Για Windows
   ```

3. **Εγκατάσταση απαιτούμενων βιβλιοθηκών:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Προσθήκη δεδομένων:**
   - Κατεβάστε το αρχείο `diabetes.csv` από το [Kaggle](https://www.kaggle.com/).
   - Τοποθετήστε το στον φάκελο `data/`.

5. **Εκπαίδευση και πρόβλεψη:**
   ```bash
   python3 src/train.py
   python3 src/predict.py
   ```

---

## 📈 Παραμετροποίηση

Για να τροποποιήσετε τις υπερπαραμέτρους, προσαρμόστε τα εξής:
- Στο `train.py`, επεξεργαστείτε τα λεξικά `param_grid_rf` και `param_grid_xgb`.
- Μπορείτε επίσης να προσθέσετε ή να αφαιρέσετε αλγορίθμους από το λεξικό `models`.

---

## ✅ Αποτελέσματα

Μετά την εκτέλεση του `train.py`, θα εμφανιστούν:
- Ακρίβεια κάθε μοντέλου
- Καλύτερες υπερπαράμετροι για Random Forest και XGBoost
- Classification Report και Confusion Matrix

Το καλύτερο μοντέλο αποθηκεύεται αυτόματα στον φάκελο `models/`.

---

## 📄 Άδειες Χρήσης

Αυτό το project διατίθεται υπό την άδεια MIT.

---

## 🙋 Συνεισφορά

Οι συνεισφορές είναι ευπρόσδεκτες! Μπορείτε να ανοίξετε ένα issue ή pull request για βελτιώσεις.


import joblib
import numpy as np

# Φόρτωση του αποθηκευμένου μοντέλου
model = joblib.load('models/diabetes_model.pkl')

# Δοκιμαστικά δεδομένα
new_data = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])

# Πρόβλεψη
prediction = model.predict(new_data)
print(f"\n🔍 Πρόβλεψη: {'Διαβήτης' if prediction[0] == 1 else 'Όχι Διαβήτης'}")

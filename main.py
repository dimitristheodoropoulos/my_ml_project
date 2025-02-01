import subprocess

# Εκπαίδευση μοντέλου
subprocess.run(["python3", "src/train.py"])

# Εκτέλεση πρόβλεψης
subprocess.run(["python3", "src/predict.py"])

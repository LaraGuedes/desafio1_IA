import joblib
import numpy as np


scaler = joblib.load("scaler_knn.joblib")

entrada = np.array([[10000, 5000, 0.98, 35]])


entrada_normalizada = scaler.transform(entrada)

print("Entrada normalizada:", entrada_normalizada)

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("elegibilidade_credito.csv") 

df['historico_pagamento (score)'] = df['historico_pagamento (score)'].str.replace('.', '', regex=False)
df['historico_pagamento (score)'] = pd.to_numeric(df['historico_pagamento (score)'], errors='coerce') / 1e17

df.dropna(inplace=True)

X = df[['salario_anual', 'total_dividas', 'historico_pagamento (score)', 'idade']]

scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X)

joblib.dump(scaler, "scaler_knn.joblib")

print("Scaler treinado e salvo como 'scaler_knn.joblib'")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


df = pd.read_csv("elegibilidade_credito.csv")


df['historico_pagamento (score)'] = df['historico_pagamento (score)'].str.replace('.', '', regex=False)
df['historico_pagamento (score)'] = pd.to_numeric(df['historico_pagamento (score)'], errors='coerce') / 1e17
df.dropna(inplace=True)


X = df[['salario_anual', 'total_dividas', 'historico_pagamento (score)', 'idade']]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)


df['cluster'] = clusters


plt.figure(figsize=(10, 6))
plt.scatter(df['salario_anual'], df['total_dividas'], c=clusters, cmap='viridis')
plt.xlabel("Salário Anual")
plt.ylabel("Total de Dívidas")
plt.title("Clustering das Solicitações de Crédito com KMeans")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

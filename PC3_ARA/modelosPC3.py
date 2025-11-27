"""
Modelos simples sobre dataset CityBike
Incluye:
- Gráficos estáticos en /graficos/*.png
"""

# -------------------------------------
# 0) Imports
# -------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import io
import os
import requests

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Configuración estética de gráficos
plt.rcParams['figure.figsize'] = (8,5)
sns.set(style="whitegrid")

# -------------------------------------
# Crear carpeta para guardar imágenes
# -------------------------------------
os.makedirs("graficos", exist_ok=True)

# -------------------------------------
# 1) Cargar dataset
# -------------------------------------
github_raw_url = ("https://raw.githubusercontent.com/"
                  "Dilan140402/Analitica-Datos/"
                  "61f05dd96a1f6e07a1721930ef267b11651f18c9/"
                  "data/citybike_lima.csv")

local_files = glob.glob("/mnt/data/*citybike*.csv") + glob.glob("/mnt/data/*city*.csv")

def load_citybike(url, local_files):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            print("CSV cargado desde GitHub.")
            return pd.read_csv(io.StringIO(r.text))
    except:
        print("Fallo descarga. Probando archivos locales...")

    for f in local_files:
        try:
            print("Cargando local:", f)
            return pd.read_csv(f)
        except:
            pass
    raise FileNotFoundError("No se encontró archivo válido.")

df = load_citybike(github_raw_url, local_files)

# -------------------------------------
# 2) Preprocesamiento básico
# -------------------------------------
df.columns = df.columns.str.lower().str.replace(" ", "_")

# detectar columna real
target_candidates = ["free_bikes", "available_bikes", "bikes_available"]
target = next((c for c in target_candidates if c in df.columns), None)

if target is None:
    raise ValueError("No se encontró columna de bicicletas disponibles")

print("TARGET:", target)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target in num_cols:
    num_cols.remove(target)

feature = num_cols[0]
print("Feature seleccionada:", feature)

df = df[[feature, target]].dropna()

X = df[[feature]].values
y = df[target].values.reshape(-1, 1)

# -------------------------------------
# 3) Train / Test
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------
# 4) MODELO 1: Ecuación Normal
# -------------------------------------
X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
y_pred_normal = X_test_b.dot(theta_best)

rmse_normal = np.sqrt(mean_squared_error(y_test, y_pred_normal))
r2_normal = r2_score(y_test, y_pred_normal)

# -------------------------------------
# 5) MODELO 2: LinearRegression
# -------------------------------------
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred_lr = lin_reg.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

# -------------------------------------
# 6) GRÁFICOS + GUARDADO PNG
# -------------------------------------

# 6.1 Ecuación Normal
plt.figure()
plt.scatter(X, y, alpha=0.4)
plt.plot(X, np.c_[np.ones((len(X), 1)), X].dot(theta_best), color="red")
plt.title("Modelo - Ecuación Normal")
plt.xlabel(feature)
plt.ylabel(target)
plt.savefig("graficos/modelo_ecuacion_normal.png")
plt.show()

# 6.2 LinearRegression
plt.figure()
plt.scatter(X, y, alpha=0.4)
plt.plot(X, lin_reg.predict(X.reshape(-1,1)), color="green")
plt.title("Modelo - LinearRegression")
plt.xlabel(feature)
plt.ylabel(target)
plt.savefig("graficos/modelo_linear_regression.png")
plt.show()

# 6.3 Comparativa RMSE
plt.figure()
plt.bar(["Ecuación Normal", "LinearRegression"], [rmse_normal, rmse_lr])
plt.title("Comparación RMSE")
plt.ylabel("RMSE")
plt.savefig("graficos/comparacion_rmse.png")
plt.show()

print("\nImágenes guardadas en carpeta → /graficos/")

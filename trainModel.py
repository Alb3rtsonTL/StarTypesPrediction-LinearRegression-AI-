# train_model.py

# Importar las bibliotecas necesarias
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib

# Cargar el conjunto de datos desde el archivo CSV
data = pd.read_csv('data\Train-DatasetStarTypes.csv.csv')

# Mostrar las primeras filas del conjunto de datos para verificar la carga
print(data.head())

# Describir el conjunto de datos
print(data.describe())

# Preparar las características (X) y la variable objetivo (Y)
X = data.drop(['Star type', 'Star color', 'Spectral Class'], axis=1)
Y = data['Star type']

# Convertir las etiquetas de la variable objetivo a valores numéricos
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)

# Convertir variables categóricas a variables numéricas y agregarlas a X
data['Star color'] = label_encoder.fit_transform(data['Star color'])
data['Spectral Class'] = label_encoder.fit_transform(data['Spectral Class'])
X = pd.concat([X, data[['Star color', 'Spectral Class']]], axis=1)

# Escalar las características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, Y_train)

# Usar el modelo para predecir
Y_pred = model.predict(X_test)

# Redondear las predicciones para que sean etiquetas de clase
Y_pred_rounded = np.round(Y_pred).astype(int)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(Y_test, Y_pred_rounded)
conf_matrix = confusion_matrix(Y_test, Y_pred_rounded)
class_report = classification_report(Y_test, Y_pred_rounded)

# Imprimir las métricas de evaluación para entender el rendimiento del modelo
print(f"Accuracy: {accuracy}\n")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Gráficas de los resultados
fig, ax = plt.subplots(2, 2, figsize=(15, 15))

# Comparación de los valores reales y predichos
ax[0, 0].plot(Y_test[:100], label='Actual')
ax[0, 0].plot(Y_pred_rounded[:100], label='Predicho')
ax[0, 0].legend()
ax[0, 0].set_title("Comparación de Valores Reales y Predichos")

# Gráfico de dispersión de los valores reales y predichos
ax[0, 1].scatter(range(len(Y_test)), Y_test, label='Actual', alpha=0.5)
ax[0, 1].scatter(range(len(Y_pred_rounded)), Y_pred_rounded, label='Predicho', alpha=0.5)
ax[0, 1].legend()
ax[0, 1].set_title("Scatter Plot de Valores Reales y Predichos")

# Histograma de los errores de predicción
ax[1, 0].hist(Y_test - Y_pred_rounded, bins=20)
ax[1, 0].set_title("Histograma de Errores de Predicción")

# Comparación de nuevos datos y predicciones
ax[1, 1].scatter(range(len(Y_pred_rounded)), Y_pred_rounded, label='Predicho', color='red')
ax[1, 1].scatter(range(len(data)), data['Star type'], label='Actual', color='blue')
ax[1, 1].legend()
ax[1, 1].set_title("Comparación de Nuevos Datos y Predicciones")

plt.tight_layout()
plt.show()

# Guardar el modelo entrenado y el escalador
joblib.dump(model, 'Model-LinearRegression.pkl')
joblib.dump(scaler, 'Scaler-LinearRegression.pkl')
joblib.dump(label_encoder, 'LabelEncoder-LinearRegression.pkl')
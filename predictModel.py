# predict.py

# Importar las bibliotecas necesarias
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import matplotlib.pyplot as plt

# Cargar el modelo, el escalador y los label_encoders
model = joblib.load('STPrediction-Model-LinearRegression.pkl')
scaler = joblib.load('STPrediction-Scaler-LinearRegression.pkl')
label_encoder = joblib.load('STPrediction-LabelEncoder-LinearRegression.pkl')

# Cargar un nuevo dataset más grande con las columnas especificadas
new_data = pd.read_csv('../data/DataSet-TestStarTypes.csv')  # Archivo CSV

# Convertir variables categóricas a variables numéricas
new_data['Star color'] = new_data['Star color'].apply(lambda x: 'Unknown' if x not in label_encoder.classes_ else x)
label_encoder.classes_ = np.append(label_encoder.classes_, 'Unknown')  # Añadir 'Unknown' si no existe
new_data['Star color'] = label_encoder.transform(new_data['Star color'])

# Convertir Spectral Class de la misma manera
new_data['Spectral Class'] = new_data['Spectral Class'].apply(lambda x: 'Unknown' if x not in label_encoder.classes_ else x)
label_encoder.classes_ = np.append(label_encoder.classes_, 'Unknown')  # Añadir 'Unknown' si no existe
new_data['Spectral Class'] = label_encoder.transform(new_data['Spectral Class'])

# Escalar las características del nuevo dataset
new_data_scaled = scaler.transform(new_data.drop(['Star type'], axis=1))

# Predecir utilizando el modelo entrenado
new_pred = model.predict(new_data_scaled)

# Redondear las predicciones para que sean etiquetas de clase
new_pred_rounded = np.round(new_pred).astype(int)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(new_data['Star type'], new_pred_rounded)
conf_matrix = confusion_matrix(new_data['Star type'], new_pred_rounded)
class_report = classification_report(new_data['Star type'], new_pred_rounded)

# Imprimir las métricas de evaluación para entender el rendimiento del modelo
print(f"Accuracy: {accuracy}\n")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Gráficas de los resultados
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# Comparación de los valores reales y predichos
ax[0, 0].plot(new_data['Star type'][:100], label='Actual')
ax[0, 0].plot(new_pred_rounded[:100], label='Predicho')
ax[0, 0].legend()
ax[0, 0].set_title("Comparación de Valores Reales y Predichos")

# Gráfico de dispersión de los valores reales y predichos
ax[0, 1].scatter(range(len(new_data['Star type'])), new_data['Star type'], label='Actual', alpha=0.5)
ax[0, 1].scatter(range(len(new_pred_rounded)), new_pred_rounded, label='Predicho', alpha=0.5)
ax[0, 1].legend()
ax[0, 1].set_title("Scatter Plot de Valores Reales y Predichos")

# Histograma de los errores de predicción
ax[1, 0].hist(new_data['Star type'] - new_pred_rounded, bins=20)
ax[1, 0].set_title("Histograma de Errores de Predicción")

# Comparación de nuevos datos y predicciones
ax[1, 1].scatter(range(len(new_pred_rounded)), new_pred_rounded, label='Predicho', color='red')
ax[1, 1].scatter(range(len(new_data)), new_data['Star type'], label='Actual', color='blue')
ax[1, 1].legend()
ax[1, 1].set_title("Comparación de Nuevos Datos y Predicciones")

plt.tight_layout()
plt.show()

# %% [markdown]
# Nombre: StarTypes-Prediction(LinearRegression-AI)
# 
# DataSet Utilizado: https://www.kaggle.com/datasets/deepu1109/star-dataset
# 
# El proyecto trata sobre predecir el tipo de estrella basado en características astronómicas utilizando el modelo de regresión lineal.
# 
# Descripción del Proyecto:
# El objetivo del proyecto es construir un modelo de regresión lineal para predecir el tipo de estrella a partir de datos astronómicos. El dataset utilizado contiene varias características de estrellas, como temperatura, luminosidad, radio, masa, entre otros, y la variable objetivo es el tipo de estrella (que puede ser una de seis clases diferentes).
# 
# By: @Alb3rtsonTL (Albertson Terrero López)


# %%
# Importar las bibliotecas necesarias
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

# %%
# Cargar el conjunto de datos desde el archivo CSV
data = pd.read_csv('data/Train-DatasetStarTypes.csv.csv')

# %%
# Mostrar las primeras filas del conjunto de datos para verificar la carga
#Se puede usar data.head() para que se vea como una tabla de excel o print(data.head()) para 
data.head()

# %%
data.describe()

# %%
# Preparar las características (X) y la variable objetivo (Y)
X = data.drop(['Star type', 'Star color', 'Spectral Class'], axis=1)
Y = data['Star type']

# %%
# Convertir las etiquetas de la variable objetivo a valores numéricos
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)

# %%
# Convertir variables categóricas a variables numéricas y agregarlas a X
data['Star color'] = label_encoder.fit_transform(data['Star color'])
data['Spectral Class'] = label_encoder.fit_transform(data['Spectral Class'])
X = pd.concat([X, data[['Star color', 'Spectral Class']]], axis=1)

# %%
# Escalar las características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# %%
# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
# Usamos el 70% de los datos para entrenamiento y el 30% para prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# %%
# Mostrar las dimensiones de los conjuntos de datos resultantes
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"Y_test shape: {Y_test.shape}")

# %% [markdown]
# Usando el modelo de regresión lineal (LinearRegression) el modelo de ia no saca su máximo potencial con este dataset solo llega a tener un accuracy de un 70% a 80%, como máximo sin mas modificaciones en el DataSet. 
# 
# Pero usando regresión logística (LogisticRegression) sube de 92% a 99% sin modificar el DataSet.

# %%
# Crear el modelo de regresión lineal
model = LinearRegression()

# %%
# Usar el modelo de regresión lógica
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(max_iter=200)

# %%
# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train,Y_train)

# %%
# Usar el modelo utilizando métricas de clasificación
Y_pred = model.predict(X_test)

# %%
model.intercept_

# %%
model.coef_

# %%
# Redondear las predicciones para que sean etiquetas de clase
Y_pred_rounded = np.round(Y_pred).astype(int)

# %%
# Evaluar el rendimiento del modelo
accuracy = accuracy_score(Y_test, Y_pred_rounded)
conf_matrix = confusion_matrix(Y_test, Y_pred_rounded)
class_report = classification_report(Y_test, Y_pred_rounded)

# Imprimir las métricas de evaluación para entender el rendimiento del modelo
print(f"Accuracy: {accuracy} \n")
print("Confusion Matrix:")
print(conf_matrix , "\n")
print("Classification Report: \n")
print(class_report)

# %%
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


# %% [markdown]
# Ahora guardo el modelo, scaler y el label_encoder para utilizar el modelo entrenado para predecir nuevos datos usando los datos para el entrenamiento y el test como base para las nuevas predicciones, para identificar los tipos de estrellas.

# %%
# Guardar el modelo entrenado y el escalador
import joblib

joblib.dump(model, 'Model-LinearRegression.pkl')
joblib.dump(scaler, 'Scaler-LinearRegression.pkl')
joblib.dump(label_encoder, 'LabelEncoder-LinearRegression.pkl')

# %%
# Procedo a cargar el modelo, el escalador y los label_encoders
model = joblib.load('Model-LinearRegression.pkl')
scaler = joblib.load('Scaler-LinearRegression.pkl')
label_encoder = joblib.load('LabelEncoder-LinearRegression.pkl')

# %%
# Cargar un nuevo dataset más grande con las columnas especificadas
new_data = pd.read_csv('data/Predict-DatasetStarTypes.csv')  # Archivo CSV

# %%
# Convertir variables categóricas a variables numéricas
new_data['Star color'] = new_data['Star color'].apply(lambda x: 'Unknown' if x not in label_encoder.classes_ else x)
label_encoder.classes_ = np.append(label_encoder.classes_, 'Unknown')  # Añadir 'Unknown' si no existe
new_data['Star color'] = label_encoder.transform(new_data['Star color'])

# %%
# Convertir Spectral Class de la misma manera
new_data['Spectral Class'] = new_data['Spectral Class'].apply(lambda x: 'Unknown' if x not in label_encoder.classes_ else x)
label_encoder.classes_ = np.append(label_encoder.classes_, 'Unknown')  # Añadir 'Unknown' si no existe
new_data['Spectral Class'] = label_encoder.transform(new_data['Spectral Class'])

# %%
# Escalar las características del nuevo dataset
new_data_scaled = scaler.transform(new_data.drop(['Star type'], axis=1))

# %%
# Predecir utilizando el modelo entrenado
new_pred = model.predict(new_data_scaled)

# %%
# Evaluar el rendimiento del modelo
accuracy = accuracy_score(Y_test, Y_pred_rounded)
conf_matrix = confusion_matrix(Y_test, Y_pred_rounded)
class_report = classification_report(Y_test, Y_pred_rounded)

# Imprimir las métricas de evaluación para entender el rendimiento del modelo
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# %%
# Gráficas de los resultados
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# Comparación de los valores reales y predichos
ax[0, 0].plot(Y_test[:100], label='Actual')
ax[0, 0].plot(Y_pred[:100], label='Predicho')
ax[0, 0].legend()
ax[0, 0].set_title("Comparación de Valores Reales y Predichos")

# Gráfico de dispersión de los valores reales y predichos
ax[0, 1].scatter(range(len(Y_test)), Y_test, label='Actual', alpha=0.5)
ax[0, 1].scatter(range(len(Y_pred)), Y_pred, label='Predicho', alpha=0.5)
ax[0, 1].legend()
ax[0, 1].set_title("Scatter Plot de Valores Reales y Predichos")

# Histograma de los errores de predicción
ax[1, 0].hist(Y_test - Y_pred, bins=20)
ax[1, 0].set_title("Histograma de Errores de Predicción")

# Comparación de nuevos datos y predicciones
ax[1, 1].scatter(range(len(new_pred)), new_pred, label='Predicho', color='red')
ax[1, 1].scatter(range(len(new_data)), new_data['Star type'], label='Actual', color='blue')
ax[1, 1].legend()
ax[1, 1].set_title("Comparación de Nuevos Datos y Predicciones")

plt.tight_layout()
plt.show()



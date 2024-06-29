# StarTypes-Prediction(LinearRegression-AI)

## Descripción del Proyecto

El objetivo del proyecto es construir un modelo de regresión lineal para predecir el tipo de estrella a partir de datos astronómicos. El dataset utilizado contiene varias características de estrellas, como temperatura, luminosidad, radio, masa, entre otros, y la variable objetivo es el tipo de estrella (que puede ser una de seis clases diferentes).

## Autor

By: @Alb3rtsonTL (Albertson Terrero López)

## Contenidos

1. [Requisitos](#requisitos)
2. [Instalación](#instalación)
3. [Uso](#uso)
4. [Entrenamiento del Modelo](#entrenamiento-del-modelo)
5. [Evaluación del Modelo](#evaluación-del-modelo)
6. [Predicción con Nuevos Datos](#predicción-con-nuevos-datos)

## Requisitos
El script `requirements.txt` realiza los siguientes pasos:

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## Instalación

1. Clona el repositorio:
    ```sh
    git clone https://github.com/Alb3rtsonTL/StarTypes-Prediction.git
    cd StarTypes-Prediction
    ```

2. Instala las dependencias:
    ```sh
    pip install -r requirements.txt
    ```

## Uso

1. Descarga el dataset desde [Kaggle](https://www.kaggle.com/datasets/deepu1109/star-dataset) y guárdalo en el directorio `data/`.

2. Corre el script principal para entrenar el modelo:
    ```sh
    python trainModel.py
    ```

3. Utiliza el modelo entrenado para hacer predicciones:
    ```sh
    python predictModel.py
    ```

## Entrenamiento del Modelo

El script `trainModel.py` realiza los siguientes pasos:

1. Cargar el conjunto de datos desde el archivo CSV.
2. Preparar las características (X) y la variable objetivo (Y).
3. Convertir las etiquetas de la variable objetivo a valores numéricos.
4. Convertir variables categóricas a variables numéricas y agregarlas a X.
5. Escalar las características.
6. Dividir el conjunto de datos en conjuntos de entrenamiento y prueba.
7. Crear y entrenar el modelo de regresión lineal.
8. Evaluar el rendimiento del modelo.

## Evaluación del Modelo

El modelo es evaluado utilizando las siguientes métricas:

- Accuracy
- Confusion Matrix
- Classification Report

Además, se generan gráficas para visualizar la comparación de valores reales y predichos, un scatter plot de valores reales y predichos, un histograma de errores de predicción y la comparación de nuevos datos y predicciones.

## Predicción con Nuevos Datos

El script `predictModel.py` realiza los siguientes pasos:

1. Cargar un nuevo dataset para hacer predicciones con las columnas especificadas.
2. Convertir variables categóricas a variables numéricas.
3. Escalar las características del nuevo dataset.
4. Predecir utilizando el modelo entrenado.
5. Evaluar el rendimiento del modelo con las nuevas predicciones.
6. Generar gráficas para visualizar los resultados de las nuevas predicciones.

## Full Code

El script `StarTypesPrediction.py` es el Notebook exportado en formato python, con el código completo.


## Guardar y Cargar el Modelo

Para guardar el modelo entrenado, el escalador y el codificador de etiquetas, se utiliza `joblib`:

```python
import joblib
joblib.dump(model, 'Model-LinearRegression.pkl')
joblib.dump(scaler, 'Scaler-LinearRegression.pkl')
joblib.dump(label_encoder, 'LabelEncoder-LinearRegression.pkl')

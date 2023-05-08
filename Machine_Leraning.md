# Ejemplo de Regresión Multilineal.

## Cargar los datos.

Instrucciones para subir el archivo "housing.csv" a tu Jupyter Notebook en Google Colab:

1. Abre tu cuenta de Google Drive en tu navegador y crea una carpeta nueva donde desees guardar tu archivo CSV. 

2. Haz clic en "Nuevo" en la esquina superior izquierda y selecciona "Más" y luego "Google Colaboratory". Esto abrirá un nuevo cuaderno de Colab en una nueva pestaña de tu navegador.

3. En el nuevo cuaderno de Colab, haz clic en "Archivo" en la barra de herramientas superior y selecciona "Subir". 

4. Busca el archivo CSV que deseas subir ("housing.csv") en tu computadora y selecciónalo.

5. Una vez que el archivo CSV se haya cargado en Colab, aparecerá en la sección de archivos en la barra lateral izquierda.

6. Puedes acceder al archivo CSV desde el código Python en tu cuaderno de Colab utilizando la ruta relativa al archivo. Por ejemplo, si colocaste el archivo CSV en una carpeta llamada "datasets", puedes leerlo en tu cuaderno de la siguiente manera:

```python
import pandas as pd

data = pd.read_csv('datasets/housing.csv')
```



## Implementación

La regresión lineal múltiple es una técnica de aprendizaje automático que permite predecir una variable objetivo (dependiente) en función de múltiples variables explicativas (independientes). En este ejemplo, utilizaremos el dataset "California Housing Prices" para predecir el precio medio de las viviendas en función de diferentes características.

El dataset "California Housing Prices" contiene información demográfica y de vivienda de los bloques censales de California en 1990. Las principales variables del dataset son:

1. Longitud: longitud geográfica del bloque censal.
2. Latitud: latitud geográfica del bloque censal.
3. Edad media de la vivienda: antigüedad promedio de las viviendas en el bloque censal.
4. Total de habitaciones: número total de habitaciones en el bloque censal.
5. Total de dormitorios: número total de dormitorios en el bloque censal.
6. Población: número de personas que viven en el bloque censal.
7. Hogares: número total de hogares en el bloque censal.
8. Ingreso medio: ingreso medio de los residentes en el bloque censal.
9. Precio medio de la vivienda: variable objetivo que queremos predecir.

A continuación, se presenta un ejemplo detallado de cómo aplicar la regresión lineal múltiple a este dataset utilizando Python y la biblioteca Scikit-learn:

1. Importamos las bibliotecas necesarias:
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

2. Cargamos el dataset y lo dividimos en características (X) y variable objetivo (y):
```python
data = pd.read_csv("datasets/housing.csv")
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]
```

3. Dividimos el dataset en conjuntos de entrenamiento y prueba (80% y 20% respectivamente):
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

4. Creamos el modelo de regresión lineal múltiple y lo ajustamos con los datos de entrenamiento:
```python
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

5. Realizamos predicciones con el conjunto de prueba y evaluamos el modelo:
```python
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

En este punto, ya hemos entrenado y evaluado el modelo de regresión lineal múltiple. Los principales descriptores estadísticos que podemos obtener son:

1. Coeficientes: representan la relación entre cada variable independiente y la variable objetivo. Cuanto mayor sea el coeficiente, mayor será el efecto de esa variable en la predicción del precio medio de la vivienda. Podemos obtenerlos con `regressor.coef_`.

2. Intercepto: es el valor de la variable objetivo cuando todas las variables independientes son iguales a cero. Se puede obtener con `regressor.intercept_`.

3. Error cuadrático medio (MSE): mide la diferencia promedio entre los valores predichos y los valores reales. Un MSE más bajo indica un mejor ajuste del modelo. Lo hemos calculado como `mse = mean_squared_error(y_test, y_pred)`.

4. Coeficiente de determinación (R²): indica qué porcentaje de la variación en la variable objetivo puede explicarse por las variables independientes. Los valores de R² oscilan entre 0 y 1, siendo 1 un ajuste perfecto. Lo hemos calculado como `r2 = r2_score(y_test, y_pred)`.

### Utilizando el modelo

En resumen, hemos utilizado la regresión lineal múltiple para predecir el precio medio de las viviendas en California basándonos en diferentes características demográficas y de vivienda. Los descriptores estadísticos nos permiten evaluar la calidad de las predicciones y la relevancia de las variables independientes en el modelo.


Una vez que hayas entrenado el modelo de regresión lineal múltiple utilizando el dataset "California Housing Prices", puedes utilizarlo para predecir el valor de una casa proporcionando los valores de las variables independientes correspondientes. Para ello, sigue estos pasos:

1. Preparar los datos de entrada: crea un array de NumPy o un DataFrame de pandas con los valores de las variables independientes de la casa que deseas predecir. Asegúrate de que los datos estén en el mismo orden que en el dataset original.

```python
# Ejemplo de datos de entrada (estos valores deben ser reemplazados por los de la casa real)
house_data = np.array([[longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income]])
```

2. Realizar la predicción utilizando el método `predict()` del modelo. Asegúrate de que el modelo ya esté entrenado con los datos de entrenamiento como se mostró en la respuesta anterior.

```python
predicted_value = regressor.predict(house_data)
```

3. Interpretar el resultado: la variable `predicted_value` contendrá un array con un solo valor, que representa la predicción del precio medio de la casa basada en las características proporcionadas.

```python
print("El precio medio de la vivienda predicho es:", predicted_value[0])
```

Recuerda que las predicciones realizadas por el modelo son aproximaciones basadas en las relaciones encontradas en el dataset de entrenamiento. La calidad de estas predicciones depende de la precisión y representatividad del modelo y de los datos.

Supongamos que queremos predecir el precio medio de una casa utilizando los datos de la primera fila del conjunto de datos. Primero, extraemos esos datos:

```python
example_house = data.iloc[0]  # Seleccionamos la primera fila del dataset
print(example_house)
```

La salida mostrará los valores de las características de la primera casa del dataset,, algo similar a esto:

```
longitude               -122.23
latitude                 37.88
housing_median_age       41.0
total_rooms             880.0
total_bedrooms          129.0
population              322.0
households              126.0
median_income             8.3252
median_house_value    452600.0
```

Ahora, preparamos los datos de entrada `house_data` extrayendo solo las variables independientes y eliminando la variable objetivo "median_house_value":

```python
house_data = example_house.drop("median_house_value").values.reshape(1, -1)
```

Finalmente, utilizamos el modelo de regresión lineal múltiple previamente entrenado para predecir el valor de esta casa:

```python
predicted_value = regressor.predict(house_data)
print("El precio medio de la vivienda predicho es:", predicted_value[0])
```

Ten en cuenta que este es solo un ejemplo y que el valor predicho puede no ser exactamente igual al valor real debido a las limitaciones del modelo y posibles errores en la predicción. Además, este ejemplo utiliza datos del propio dataset, pero en aplicaciones reales, generalmente se usarían datos de casas no presentes en el dataset para realizar predicciones útiles y nuevas.
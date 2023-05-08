# Ejemplo de Regresión Linea.

## Configurando el cuaderno de Jupyter

Los pasos para crear un nuevo cuaderno Jupyter en Google Colab:

1. Abre tu navegador web y ve a la página principal de Google Colab: https://colab.research.google.com/
2. Si no has iniciado sesión en tu cuenta de Google, haz clic en "Iniciar sesión" y sigue los pasos para iniciar sesión.
3. Una vez que hayas iniciado sesión, haz clic en "Nuevo cuaderno" en la esquina superior izquierda de la página.
4. Se abrirá una ventana emergente que te permitirá seleccionar el tipo de cuaderno que deseas crear. Puedes elegir entre "Python 3", "R" o "Notebook vacío". También puedes seleccionar "Abrir un cuaderno existente" para abrir un archivo guardado anteriormente.
5. Después de elegir la opción deseada, se abrirá un nuevo cuaderno en una pestaña nueva del navegador. Puedes empezar a escribir código y notas en las celdas del cuaderno.

## Regresión lineal con datos artificiales.


Para esta demostración, vamos a trabajar con un conjunto de datos simulados que contiene dos variables: X y Y. El objetivo es encontrar la ecuación de la línea de regresión que mejor describe la relación entre las dos variables.

Primero, importamos las bibliotecas necesarias:

``` python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

A continuación, creamos nuestro conjunto de datos simulados. En este caso, utilizaremos la función `np.random.randn()` de NumPy para generar dos arreglos de datos aleatorios, uno para X y otro para Y:

``` python
# Crear un conjunto de datos simulados
np.random.seed(0)  # Para reproducibilidad
x = np.random.randn(50)
y = x * 2 + np.random.randn(50)
```

En este ejemplo, estamos generando 50 puntos de datos aleatorios para X, y luego calculando los valores correspondientes de Y. La relación entre X y Y está definida por `y = 2x + ruido`, donde `ruido` es una pequeña cantidad de ruido aleatorio agregado para hacer los datos más realistas.

Podemos visualizar estos datos utilizando un diagrama de dispersión simple. Para hacerlo, usamos la función `plt.scatter()` de Matplotlib:

``` python
# Diagrama de dispersión de los datos
plt.scatter(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

A continuación, calculamos los coeficientes de la regresión lineal simple utilizando la fórmula de mínimos cuadrados. Para hacer esto, primero necesitamos calcular la media de X y Y:

``` python
# Calcular la media de X e Y
x_mean = np.mean(x)
y_mean = np.mean(y)
```

Luego, podemos calcular los términos `b1` y `b0` de la ecuación de la línea de regresión utilizando las siguientes fórmulas:

``` python
# Calcular los coeficientes de la regresión lineal
numerator = 0
denominator = 0
for i in range(len(x)):
    numerator += (x[i] - x_mean) * (y[i] - y_mean)
    denominator += (x[i] - x_mean) ** 2
b1 = numerator / denominator
b0 = y_mean - b1 * x_mean
```

Ahora que hemos calculado los coeficientes de la regresión lineal, podemos trazar la línea de regresión en el diagrama de dispersión utilizando la función `plt.plot()` de Matplotlib. Primero, generamos una serie de valores de X que abarquen el rango de los datos originales, y luego calculamos los valores correspondientes de Y utilizando la ecuación de la línea de regresión:

``` python
# Trazar la línea de regresión en el diagrama de dispersión
x_range = np.linspace(np.min(x), np.max(x), 10)
y_range = b0 + b1 * x_range
plt.plot(x_range, y_range, color='red')
plt.scatter(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

Finalmente, podemos calcular el coeficiente de determinación R², que es una medida de cuánto de la variabilidad en Y se explica por la regresión lineal. Podemos calcular R² utilizando la siguiente fórmula:

``` python
# Calcular el coeficiente de determinación R²
ss_tot = 0
ss_res = 0
for i in range(len(x)):
    y_pred = b0 + b1 * x[i]
    ss_tot += (y[i] - y_mean) ** 2
    ss_res += (y[i] - y_pred) ** 2
r_squared = 1 - (ss_res / ss_tot)
```

En este caso, el valor de R² es aproximadamente 0,74, lo que significa que alrededor del 74% de la variabilidad en Y se explica por la regresión lineal con respecto a X.

Una vez que has calculado los coeficientes de la regresión lineal simple, puedes utilizar la ecuación de la línea de regresión para predecir el valor de Y correspondiente a un valor de X específico. La ecuación de la línea de regresión es:

`y = b0 + b1 * x`

Donde `b0` y `b1` son los coeficientes de la regresión lineal que hemos calculado previamente.

Para predecir el valor de Y correspondiente a un valor de X específico, simplemente sustituye el valor de X en la ecuación de la línea de regresión y realiza el cálculo correspondiente. Por ejemplo, si quisieras predecir el valor de Y correspondiente a X = 1, puedes hacer lo siguiente:

``` python
# Calcular la predicción de Y para X = 1
x_pred = 1
y_pred = b0 + b1 * x_pred
print("La predicción de Y para X = 1 es:", y_pred)
```

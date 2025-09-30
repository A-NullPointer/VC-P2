<h1 align="center">Práctica 1</h1>

<h2 align="center">Asignatura: Visión por Computador</h2>

Universidad de Las Palmas de Gran Canaria  
Escuela de Ingeniería en Informática  
Grado de Ingeniería Informática  
Curso 2025/2026 

<h2 align="center">Autores</h2>

- Asmae Ez Zaim Driouch
- Javier Castilla Moreno

<h2 align="center">Bibliotecas utilizadas</h2>

[![NumPy](https://img.shields.io/badge/NumPy-%23013243?style=for-the-badge&logo=numpy)](https://numpy.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-%23FD8C00?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%43FF6400?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Pillow](https://img.shields.io/badge/Pillow-%23000000?style=for-the-badge&logo=pillow)](https://pypi.org/project/pillow/)


## Cómo usar
### Primer paso: clonar este repositorio
```bash
git clone "https://github.com/A-NullPointer/VC-P2"
```
### Segundo paso: Activar tu envinroment e instalar dependencias
> [!NOTE]
> Todas las dependencias pueden verse en [este archivo](envinronment.yml). Si se desea, puede crearse un entorno de Conda con dicho archivo.

Si se opta por crear un nuevo `Conda envinronment` a partir del archivo expuesto, es necesario abrir el `Anaconda Prompt` y ejecutar lo siguiente:

```bash
conda env create -f environment.yml
```

Posteriormente, se activa el entorno:

```bash
conda activate VC_P2
```

### Tercer paso: ejecutar el cuaderno
Finalmente, abriendo nuestro IDE favorito y teniendo instalado todo lo necesario para poder ejecutar notebooks, se puede ejecutar el cuaderno de la práctica [Practica2.ipynb](Practica2.ipynb) seleccionando el envinronment anteriormente creado.

> [!IMPORTANT]
> Todos los bloques de código deben ejecutarse en órden, de lo contrario, podría ocasionar problemas durante la ejecución del cuaderno.

<h1 align="center">Tareas</h1>

<h2 align="center">Tarea 1: Contar píxeles no nulos en cada fila haciendo un conteo de aquellas que tienen un valor obtenido mayoor o igual al 90% del máximo</h2>

<h2 align="center">Tarea 2: Aplicar umbralizado a imagen resulante de sobel. Posteriormente, contar filas y columnas con píxeles no nulos, remarcando aquellas que tengan un valor mayor al 90% del máximo. Canny vs Sobel</h2>

Para realizar esta tarea, se ha usado la misma imagen del mandril leída de disco anteriormente para posteriormente aplicar Sobel y a continuación umbralizar la imagen.

```python
gaussian_image = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (3, 3), 0)
sobel_y = cv2.Sobel(gaussian_image, cv2.CV_64F, 0, 1)
sobel_x = cv2.Sobel(gaussian_image, cv2.CV_64F, 1, 0)
sobel = cv2.convertScaleAbs(cv2.add(sobel_x, sobel_y))
```

Primero se ha aplicado un filtro Gaussiano, que nos difuminará ligeramente la imagen para posteriormente, aplicar Sobel en vertical y horizontal. Finalmente se juntan los resultados y se obtiene la imagen de Sobel con bordes verticales y horizontales

A continuación se muestra el resultado:

<table align="center">
   <td>
      <h3 align="center">Sobel horizontal</h3>
      <img src="imgs/mandril_sobel_horizontal.jpg">
   </td>
   <td>
      <h3 align="center">Sobel vertical</h3>
      <img src="imgs/mandril_sobel_vertical.jpg">
   <td>
      <h3 align="center">Sobel combinado</h3>
      <img src="imgs/mandril_sobel_solo.jpg">                                             
</table>

Posteriormente, para obtener el umbral necesario, se ha calculado el histograma de esta imagen a la que se le ha aplicado Sobel con ayuda de la función `cv2.calcHist` de OpenCV:

```python
histogram = cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)], [0], None, [256], [0, 256])
```

Y este es el histograma resultante:

<img src="imgs/mandril_histograma.jpg">

En el valle situado entre los dos picos del histograma, podemos encontrar el umbral necesario.

> [!NOTE]
> Puede sumarse `cv2.THRES_OTSU` al realizar el umbralizado con ayuda de OpenCV para obtener un umbral de manera automática. Posteriormente será utilizado en el demostrador. Sin embargo, en este punto se ha decidio utilizar un valor seleccionado a ojo en ese valle.

Una vez obtenemos el umbral que necesitábamos, aplicamos el umbralizado a la imagen de sobel con la función `cv2.thresold`:

```python
threshold = 130

_, threshold_image = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), threshold, 255, cv2.THRESH_BINARY)
```

El resultado obtenido es el siguiente:

<img src="imgs/mandril_umbralizado.jpg">

Tras umbralizar la imagen, podemos seguir con la tarea. Ahora, se intenta contar las filas y columnas con valores no nulos de la imagen resultante para posteriormente, marcar en la misma aquellas que superen el 90% del máximo de píxeles encontrados en una fila / columna.

Para lograr lo descrito anteriormente, se ha recurrido a la función `np.count_nonzero` de Numpy, la cual es perfecta para esta situación, pues nos dará un array donde cada valor corresponde al número de píxeles no nulos en dicha fila / columna. La función desarrollada para tal fin es la siguiente:

```python
def count_above(image, p, axis):
    counts = np.count_nonzero(image, axis=axis)
    max_index = np.argmax(counts)
    max_value = counts[max_index]
    return {i: int(counts[i]) for i in range(len(counts)) if counts[i] > p*max_value}
```

A la función le pasámos como parámetros la propia imagen, el porcentaje objetivo y el eje (filas / columnas).

> [!NOTE]
> Para encontrar la posición y el valor máximo se ha hecho uso de la función `np.argmax`de Numpy, la cual devuelve el índice del elemento con el valor máximo dentro del array.

Seguidamente, hacemos el conteo de aquellas filas y columnas con valores por encima del 90% del máximo:

```python
rows = count_above(threshold_image, .9, 1)
columns = count_above(threshold_image, .9, 0)
```

Tras obtener estos datos, podemos remarcar dichos ejes que cumplan la condición planteada en esta tarea usando primitivas gráficas de OpenCV del siguiente modo:

```python
h, w = threshold_image.shape
threshold_image = cv2.cvtColor(threshold_image, cv2.COLOR_GRAY2RGB)

for row, value in rows.items():
    cv2.line(threshold_image, (0, row), (w, row), (255, 0, 0), 1)

for col, value in columns.items():
    cv2.line(threshold_image, (col, 0), (col, h), (0, 255, 0), 1)
```

A continuación se muestra el resultado:

<img src="imgs/mandril_umbralizado_filas_columnas.jpg">

Finalmente, lo compararemos con Canny. Se han seguido los mismo procedimientos que en el caso de Sobel y este es el resultado:

<img src="imgs/mandril_canny_vs_sobel.jpg" align="center">

Como se puede observar, en el caso de Canny, el umbralizado prácticamente no ha hecho efecto. Esto se debe a que Canny no termina de encerrar áreas que luego serán encerradas por el umbralizado como bien hace Sobel, es por ello que los píxeles no nulos por columnas y filas disminuyen considerablemente.

<h2 align="center">Tarea 3: Demostrador</h2>

Para el desarrollo de este demostrador, se ha reutilizado la clase desarrollada en la práctica anterior para aplicar transformaciones a los canales de una imagen, ampliando sus métodos para añadir nuevas formas de modificar la imagen con conceptos aprendidos en estas dos prácticas. Además, se ha englobado todo ello en una pequeña aplicación usando la biblioteca gráfica de python, `Tkinter`.

Los nuevos métodos son los siguientes:

```python
```

En concreto, se han añadido utilizades para:
- Hacer un collage con inversión completa de los colores de la imagen e inversión individual de cada canal de la misma
- Aplicar Canny a cada fotograma e invertir la misma, consiguiendo un efecto de dibujo
- Resaltar los objetos con valores HSV seleccionados, dejando el resto de la imagen en escala de grises
- Aplicar Sobel a cada fotograma
- Aplicar Sobel y posterior umbralizado a cada fotograma

> [!NOTE]
> Los controles de la aplicación pueden verse en la misma ventana de ejecución. El modo actual será mostrado justo encima del vídeo.

A continuación, se muestran todos los modos disponibles en funcionamiento:

<table align="center">
    <img src="">
    <img src="">
    <img src="">
    <img src="">
    <img src="">
    <img src="">
    <img src="">
    <img src="">
</table>

<h2 align="center">Tarea 4: Reinterpretación de Air Guitar</h2>

<h2 align="center">Bibliografía</h2>

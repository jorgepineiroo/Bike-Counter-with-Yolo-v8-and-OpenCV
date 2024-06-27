<a name="br1"></a> 

Trabajo Imagen: Contador de Bicicletas

Jorge Ángel Piñeiro Acevedo

Objetivos:

Realizar un programa utilizando técnicas de *deep learning* para conseguir, a partir de un

vídeo identiﬁcar y contar las bicicletas que aparecen.

Procedimiento:

Después de investigar un poco decidí utilizar el algoritmo *YOLO,* muy popular en el campo

de la visión artiﬁcial que se emplea especialmente para detección de objetos y

segmentación.

Primero decidí entrenar un modelo utilizando *datasets* públicos, que obtuve en *Kaggle*, sin

embargo, los resultados obtenidos no eran muy buenos, ya que detectaba bicicletas, pero

cuando había gente sobre ellas se confundía y solía detectar una bicicleta por cada rueda,

como se puede ver en las imágenes:

Hice varias pruebas cambiando los parámetros de entrenamiento como las épocas o el

*batch*, pero el problema no parecía resolverse, así que decidí crear y etiquetar mi propio

*dataset* con el ﬁn de obtener mejores resultados.

Para ello descargué todas las fotos que pude de internet en las que aparecía gente

montada en bicicleta, intenté que fueran lo más variadas posible, es decir incluyendo

todos los tipos de bicicleta y también calidad de las imágenes y condiciones de

exposición, saturación, contraste…



<a name="br2"></a> 

Además, para conseguir fotos con mayor calidad salí con mi cámara por las calles de

Valencia y me coloqué cerca de los carriles bici más transitados para poder fotograﬁar la

mayor cantidad de bicis posible, esta vez también intenté que hubiera la mayor variedad

de condiciones con el ﬁn de obtener un *dataset* de mejor calidad.

Finalmente obtuve un total de 353 fotos, utilizando la herramienta *Roboﬂow*, las etiqueté

señalando con un recuadro dónde estaban las bicicletas.

A su vez utilizando *Roboﬂow*, pude aplicar el procedimiento de aumento de datos a mi

*dataset*, es decir pude triplicar la cantidad de datos aplicando transformaciones a las

imágenes, tales como rotaciones, escalado, cambios de saturación y exposición…



<a name="br3"></a> 

Una vez hecho esto dividí las imágenes en 3 grupos: imágenes de entrenamiento (train),

destinadas exclusivamente para entrenar mi modelo; imágenes de validación (validation),

utilizadas para evitar el *overﬁtting* y ajustar los hiperparámetros y por último imágenes de

prueba (test) utilizadas para evaluar el rendimiento del modelo una vez ﬁnaliza cada época

de entrenamiento.

Una vez completé mi *dataset*, pude descargarlo en un formato adecuado para utilizar

YOLOv8, con carpetas con las imágenes divididas y con las etiquetas en formato .txt con la

información de los recuadros donde se encontraban las bicicletas.

Entrenamiento:

Para entrenar mi modelo utilicé *Google Colab* ya que me permite conectarme a una GPU

más potente y puedo hacer el procedimiento mucho más rápido.

A continuación, está el código utilizado:

La librería *ultralytics* nos dará acceso al modelo de detección YOLO mencionado

anteriormente.

#Instalamos la librería ultralytics

!pip install ultralytics

Utilizamos el siguiente comando para utilizar los archivos que tenemos guardados en

nuestro *Google Drive*, por ejemplo, nuestro *dataset*.

\# Montar Google Drive

from google.colab import drive

drive.mount('/content/drive')



<a name="br4"></a> 

Al ejecutar el siguiente comando procedemos a entrenar nuestro modelo, ordenamos que

lo entrene para detección, que utilice el modelo yolov8n, la n quiere decir “nano” una

versión del modelo con menos capas y parámetros con el ﬁn de ser más ligera y rápida.

Especiﬁcamos la ruta a nuestro *dataset* y también la ruta donde guardará los resultados

del entrenamiento.

En este caso y tras hacer varias pruebas, obtuve mejores resultados ajustando las épocas

a 20 y el *batch* a 16 imágenes.

!yolo task=detect mode=train model=yolov8n.pt data=

"/content/drive/MyDrive/Detector de bicicletas/Dataset\_Jorge/data.yaml"

epochs=20 imgsz=800 batch=16 project=

"/content/drive/MyDrive/yolov8/training\_results" name=bicycle\_v8

Una vez ﬁnalizado el entrenamiento se generará un archivo *.pt* (*PyTorch*) que contendrá los

pesos de la red neuronal de nuestro modelo.

Programa:

A continuación, explicaré el código programado con el ﬁn de contar las bicicletas en un

vídeo, utilizando la biblioteca *OpenCv* y los pesos generados durante el entrenamiento.

De forma similar a antes importamos las librerías necesarias:

from ultralytics import YOLO

import cv2

Guardamos el vídeo en una variable y veriﬁcamos que se abre correctamente:

\# Guardamos el vídeo en una variable

cap = cv2.VideoCapture(ruta)

\# Verificar si el video se ha abierto correctamente

if not cap.isOpened():

print("Error: No se puede abrir el video.")



<a name="br5"></a> 

Importamos nuestro modelo entrenado previamente e inicializamos variables necesarias

más adelante, tales como el umbral de conﬁanza o la longitud de *frames* del historial.

modelo = YOLO("pesos\_dataset\_jorge.pt")

\# Clases

classNames = ["bicicleta"]

\# Umbral de confianza

umbral\_confianza= 0.65

\# Contador de bicicletas

bike\_count = 0

\# Historial de posiciones para rastreo más suave

historial = []

longitud\_h = 10 # número de frames para mantener en el historial

umbral\_iou = 0.15 # Umbral IoU para considerar la misma bicicleta

Deﬁnimos la función para calcular la intersección sobre la unión (IOU). Para calcular este

valor toma los parámetros las cajas delimitadoras de dos *frames* diferentes, es decir las

coordenadas de la esquina superior izquierda y la esquina inferior derecha de la cada caja.

A partir de estos datos se calcula el área de intersección y el área de cada una de las cajas

para ﬁnalmente calcular y devolver el valor de la intersección sobre la unión que valdrá 1 si

las cajas están exactamente en la misma posición y 0 si no hay intersección entre ellas.

#Función para calcular la relación de interseccion sobre la unión

def calcular\_iou(boxA, boxB):

xA = max(boxA[0], boxB[0])

yA = max(boxA[1], boxB[1])

xB = min(boxA[2], boxB[2])

yB = min(boxA[3], boxB[3])

interArea = max(0, xB - xA + 1) \* max(0, yB - yA + 1) #Área de intersección

boxAArea = (boxA[2] - boxA[0] + 1) \* (boxA[3] - boxA[1] + 1) #Área del primer

cuadro delimitador

boxBArea = (boxB[2] - boxB[0] + 1) \* (boxB[3] - boxB[1] + 1) #Área del

segundo cuadro delimitador

iou = interArea / float(boxAArea + boxBArea - interArea) #Relación de

intersección sobre unión

return iou



<a name="br6"></a> 

Ahora inicializaremos el bucle que recorrerá el vídeo frame a frame y detectará las

bicicletas que encuentre en las imágenes y guardamos el fotograma en una variable:

while cap.isOpened():

try:

success, img = cap.read()

if not success:

break

#Si el vídeo no se abre, interrumpir bucle

Aumentamos la exposición para evitar errores:

img = cv2.convertScaleAbs(img, alpha=1.15) #Aumentar ligeramente la exposición

para mejorar los resultados

Pasamos la imagen nuestro modelo para detectar las bicicletas y guardamos los

resultados en una variable:

resultados = modelo(img, stream=True) # Pasar cada frame por el modelo de

detección de bicicletas

Inicializamos la variable en la que guardar las coordenadas de las cajas de las bicicletas

detectadas en ese frame:

actual\_positions = []

Inicializamos el bucle que itera sobre los resultados obtenidos por nuestro modelo, que

contendrá la información de las bicicletas detectadas en la imagen, utilizando la OpenCv

se dibuja un recuadro sobre cada bicicleta si se supera el umbral de conﬁanza deﬁnido.

for r in resultados:

#Itera sobre la colección de resultados de

detección de objetos de cada frame del vídeo

boxes = r.boxes

objetos detectados en el vídeo

for box in boxes:

#Extrae la información de cada uno de los

x1, y1, x2, y2 = box.xyxy[0]

\# Extrae las coordenadas de la

esquina superior izquierda y la esquina inferior derecha de la caja delimitadora

x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) #Las

convierte a enteros

confianza = box.conf[0]

de la detección del objeto

#Extrae el nivel de confianza



<a name="br7"></a> 

if confianza < umbral\_confianza: # Verificar si la confianza de

la detección es menor que el umbral de confianza

continue

cls = int(box.cls[0])

#Extrae la clase del objeto detectado

y lo convierte a entero

actual\_positions.append((x1, y1, x2, y2)) #Añadir las coordenadas

de la caja delimitadora de la bicicleta detectada a la lista

\# Dibujar caja en el video

cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

\# Detalles del objeto

org = (x1, y1)

font = cv2.FONT\_HERSHEY\_SIMPLEX

fontScale = 1

color = (255, 0, 0)

thickness = 2

cv2.putText(img, classNames[cls], org, font, fontScale, color,

thickness) #Dibuja el texto en el vídeo sobre la caja.

A continuación, se comparan las posiciones de las bicicletas detectadas con el historial y

se calcula el valor de la intersección sobre la unión (IOU), si supera el umbral de IOU

deﬁnido previamente considerará que la bicicleta es la misma y no detectará una bicicleta

nueva, ﬁnalmente actualiza el historial de posiciones:

\# Comparar posiciones actuales con las del historial

for current\_pos in actual\_positions: #Itera sobre cada bicicleta

detectada en el frame

new\_bike = True

#Comparar la posición actual de la bicicleta (current\_pos) con cada

posición anterior (prev\_pos) en el historial.

for history\_frame in historial:

for prev\_pos in history\_frame:

iou = calcular\_iou(current\_pos, prev\_pos) #Calcula la

intersección de los cuadros delimitadores

if iou > umbral\_iou:

#Si es mayor al

umbral no detectará una nueva bicicleta

new\_bike = False

break

if not new\_bike:

break



<a name="br8"></a> 

\# Si new\_bike sigue siendo True después de comparar con todo el

historial, se incrementa el contador de bicicletas.

if new\_bike:

bike\_count += 1

\# Actualizar el historial de posiciones

historial.append(actual\_positions)

if len(historial) > longitud\_h:

historial.pop(0)

Finalmente, muestra en contador de bicicletas en la pantalla y establece un comando de

salida para interrumpir la reproducción.

\# Mostrar contador de bicicletas

cv2.putText(img, f"Numero de bicis: {bike\_count}", (15, 460),

cv2.FONT\_HERSHEY\_TRIPLEX, 1, (0, 255,0), 2)

cv2.imshow('Video', img)

if cv2.waitKey(1) == ord('q'):

break

except Exception as e:

print(f"Error durante el procesamiento del frame: {e}")

break

cap.release()

cv2.destroyAllWindows()



<a name="br9"></a> 

Resultados:

Como podremos observar el programa tiene un rendimiento bastante bueno incluso si las

condiciones de luz no son las ideales, pudiendo detectar y contar bicicletas en vídeos con

mucho contraste entre luces y sombras.

•

Vídeo 1:

•

Vídeo 2:



<a name="br10"></a> 

•

Vídeo 3:

•

Vídeo 4:



<a name="br11"></a> 

Bibliografía:

•

•

<https://docs.ultralytics.com/>

[How](https://www.youtube.com/watch?v=ZzC3SJJifMg)[ ](https://www.youtube.com/watch?v=ZzC3SJJifMg)[to](https://www.youtube.com/watch?v=ZzC3SJJifMg)[ ](https://www.youtube.com/watch?v=ZzC3SJJifMg)[train](https://www.youtube.com/watch?v=ZzC3SJJifMg)[ ](https://www.youtube.com/watch?v=ZzC3SJJifMg)[YOLOv8](https://www.youtube.com/watch?v=ZzC3SJJifMg)[ ](https://www.youtube.com/watch?v=ZzC3SJJifMg)[Object](https://www.youtube.com/watch?v=ZzC3SJJifMg)[ ](https://www.youtube.com/watch?v=ZzC3SJJifMg)[Detection](https://www.youtube.com/watch?v=ZzC3SJJifMg)[ ](https://www.youtube.com/watch?v=ZzC3SJJifMg)[on](https://www.youtube.com/watch?v=ZzC3SJJifMg)[ ](https://www.youtube.com/watch?v=ZzC3SJJifMg)[Custom](https://www.youtube.com/watch?v=ZzC3SJJifMg)[ ](https://www.youtube.com/watch?v=ZzC3SJJifMg)[Dataset](https://www.youtube.com/watch?v=ZzC3SJJifMg)[ ](https://www.youtube.com/watch?v=ZzC3SJJifMg)[|](https://www.youtube.com/watch?v=ZzC3SJJifMg)[ ](https://www.youtube.com/watch?v=ZzC3SJJifMg)[step](https://www.youtube.com/watch?v=ZzC3SJJifMg)[ ](https://www.youtube.com/watch?v=ZzC3SJJifMg)[by](https://www.youtube.com/watch?v=ZzC3SJJifMg)[ ](https://www.youtube.com/watch?v=ZzC3SJJifMg)[step](https://www.youtube.com/watch?v=ZzC3SJJifMg)[ ](https://www.youtube.com/watch?v=ZzC3SJJifMg)[Tutorial](https://www.youtube.com/watch?v=ZzC3SJJifMg)[ ](https://www.youtube.com/watch?v=ZzC3SJJifMg)[|](https://www.youtube.com/watch?v=ZzC3SJJifMg)

[Google](https://www.youtube.com/watch?v=ZzC3SJJifMg)[ ](https://www.youtube.com/watch?v=ZzC3SJJifMg)[Colab](https://www.youtube.com/watch?v=ZzC3SJJifMg)

•

•

[Real-time](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)[ ](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)[Object](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)[ ](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)[Detection](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)[ ](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)[with](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)[ ](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)[YOLO](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)[ ](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)[and](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)[ ](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)[Webcam:](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)[ ](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)[Enhancing](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)[ ](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)[Your](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)[ ](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)[Computer](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)

[Vision](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)[ ](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)[Skills](https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)

[Dataset](https://hydra.ojack.xyz/?sketch_id=flor_0)[ ](https://hydra.ojack.xyz/?sketch_id=flor_0)[de](https://hydra.ojack.xyz/?sketch_id=flor_0)[ ](https://hydra.ojack.xyz/?sketch_id=flor_0)[Kaggle](https://hydra.ojack.xyz/?sketch_id=flor_0)[ ](https://hydra.ojack.xyz/?sketch_id=flor_0)[con](https://hydra.ojack.xyz/?sketch_id=flor_0)[ ](https://hydra.ojack.xyz/?sketch_id=flor_0)[el](https://hydra.ojack.xyz/?sketch_id=flor_0)[ ](https://hydra.ojack.xyz/?sketch_id=flor_0)[que](https://hydra.ojack.xyz/?sketch_id=flor_0)[ ](https://hydra.ojack.xyz/?sketch_id=flor_0)[obtuve](https://hydra.ojack.xyz/?sketch_id=flor_0)[ ](https://hydra.ojack.xyz/?sketch_id=flor_0)[malos](https://hydra.ojack.xyz/?sketch_id=flor_0)[ ](https://hydra.ojack.xyz/?sketch_id=flor_0)[resultados](https://hydra.ojack.xyz/?sketch_id=flor_0)


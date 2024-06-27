# Contador-de-bicicletas-con-YOLOv8
Se trata de un contador de bicicletas en un vídeo, el programa es capaz de contar e identificar en la imagen las bicicletas que pasan usando herramientas de deep learning como YOLOv8
### Entrenamiento
Para entrenar mi modelo utilicé Google Colab ya que me permite conectarme a una GPU más potente y puedo hacer el procedimiento mucho más rápido.  
A continuación, está el código utilizado:

La librería ultralytics nos dará acceso al modelo de detección YOLO mencionado anteriormente.  
```python
!pip install ultralytics
````
Utilizamos el siguiente comando para utilizar los archivos que tenemos guardados en nuestro Google Drive, por ejemplo, nuestro dataset.  
```python
google.colab import drive  
drive.mount('/content/drive')
````

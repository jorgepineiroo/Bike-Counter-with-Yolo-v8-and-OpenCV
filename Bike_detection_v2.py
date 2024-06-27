# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 20:17:47 2024

@author: jorge
"""

from ultralytics import YOLO
import cv2

# Ruta del video
ruta = "video1.mp4"

# Guardamos el vídeo en una variable
cap = cv2.VideoCapture(ruta)

# Verificar si el video se ha abierto correctamente
if not cap.isOpened():
    print("Error: No se puede abrir el video.")


modelo = YOLO("pesos_dataset_jorge.pt")

# Clases
classNames = ["bicicleta"]

# Umbral de confianza
umbral_confianza= 0.7

# Contador de bicicletas
bike_count = 0

# Historial de posiciones para rastreo más suave
historial = []
longitud_h = 10  # número de frames para mantener en el historial
umbral_iou = 0.15  # Umbral IoU para considerar la misma bicicleta

#Función para calcular la relación de interseccion sobre la unión
def calcular_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1) #Área de intersección

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1) #Área del primer cuadro delimitador
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1) #Área del segundo cuadro delimitador

    iou = interArea / float(boxAArea + boxBArea - interArea) #Relación de intersección sobre unión 
    return iou

while cap.isOpened():
    try:
        success, img = cap.read() 
        if not success:
            break               #Si el vídeo no se abre, interrumpir bucle
        
        img = cv2.convertScaleAbs(img, alpha=1.15) #Aumentar ligeramente la exposición para mejorar los resultados
        
        resultados = modelo(img, stream=True)  # Pasar cada frame por el modelo de detección de bicicletas

        actual_positions = []

        for r in resultados:           #Itera sobre la colección de resultados de detección de objetos de cada frame del vídeo
            boxes = r.boxes         #Extrae la información de cada uno de los objetos detectados en el vídeo
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]      # Extrae las coordenadas de la esquina superior izquierda y la esquina inferior derecha de la caja delimitadora
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)   #Las convierte a enteros

                confianza = box.conf[0]             #Extrae el nivel de confianza de la detección del objeto
                if confianza < umbral_confianza:   # Verificar si la confianza de la detección es menor que el umbral de confianza
                    continue

                cls = int(box.cls[0])      #Extrae la clase del objeto detectado y lo convierte a entero
                actual_positions.append((x1, y1, x2, y2)) #Añadir las coordenadas de la caja delimitadora de la bicicleta detectada a la lista

                # Dibujar caja en el video
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Detalles del objeto
                org = (x1, y1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)  #Dibuja el texto en el vídeo sobre la caja.

        # Comparar posiciones actuales con las del historial
        for current_pos in actual_positions:    #Itera sobre cada bicicleta detectada en el frame
            new_bike = True
            
            #Comparar la posición actual de la bicicleta (current_pos) con cada posición anterior (prev_pos) en el historial.
            for history_frame in historial:
                for prev_pos in history_frame:
                    iou = calcular_iou(current_pos, prev_pos)   #Calcula la intersección de los cuadros delimitadores
                    if iou > umbral_iou:                        #Si es mayor al umbral no detectará una nueva bicicleta
                        new_bike = False
                        break
                if not new_bike:
                    break
                
            # Si new_bike sigue siendo True después de comparar con el historial, se incrementa el contador de bicicletas.
            if new_bike:
                bike_count += 1

        # Actualizar el historial de posiciones
        historial.append(actual_positions)
        if len(historial) > longitud_h:
            historial.pop(0)

        # Mostrar contador de bicicletas
        cv2.putText(img, f"Numero de bicis: {bike_count}", (15, 460), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255,0), 2)

        cv2.imshow('Video', img)
        if cv2.waitKey(1) == ord('q'):
            break
    except Exception as e:
        print(f"Error durante el procesamiento del frame: {e}")
        break

cap.release()
cv2.destroyAllWindows()







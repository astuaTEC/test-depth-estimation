import cv2
import numpy as np
import tensorflow as tf
import time
from fps_limiter import FPSCounter

# Cargar el modelo TFLite
interpreter = tf.lite.Interpreter(model_path="./models/modelo-5.tflite")
interpreter.allocate_tensors()

# Obtener las dimensiones esperadas de entrada para el modelo
input_details = interpreter.get_input_details()[0]
HEIGHT, WIDTH = input_details['shape'][1], input_details['shape'][2]
HEIGHT2, WIDTH2 = 480, 640

# Inicializar la cámara web
cap = cv2.VideoCapture(0)  # 0 para la cámara predeterminada, puedes cambiarlo según tu configuración
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cv2.namedWindow("Input and Depth Map", cv2.WINDOW_NORMAL)

# Generar un nombre único para el archivo de salida
current_time = time.strftime("%Y%m%d-%H%M%S")
output_filename = f'./video/output_{current_time}.mp4'

# Configuración para guardar el video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, 6, (2 * WIDTH2, HEIGHT2))

# Inicializar el contador de FPS
fps_counter = FPSCounter()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Redimensionar la imagen a las dimensiones esperadas por el modelo
    image = cv2.resize(frame, (WIDTH, HEIGHT))
    input_data = np.expand_dims(image, axis=0)
    input_data = tf.image.convert_image_dtype(input_data, tf.float32)

    # Realizar la inferencia utilizando el modelo TFLite
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    pred_depth_map = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

    # Normalizar el mapa de profundidad para mostrarlo correctamente
    pred_depth_map = (pred_depth_map - pred_depth_map.min()) / (pred_depth_map.max() - pred_depth_map.min())

    # Convertir el mapa de profundidad en una imagen en escala de grises
    pred_depth_map = (pred_depth_map * 255).astype(np.uint8)

    # Redimensionar el mapa de profundidad a las dimensiones de la imagen de entrada
    pred_depth_map_resized = cv2.resize(pred_depth_map.squeeze(), (WIDTH2, HEIGHT2))

    # Convertir el mapa de profundidad redimensionado a una imagen a color para mostrarlo junto a la imagen de entrada
    pred_depth_map_resized_inverted = cv2.bitwise_not(pred_depth_map_resized)
    pred_depth_map_colored = cv2.applyColorMap(pred_depth_map_resized_inverted, cv2.COLORMAP_MAGMA)

    # Combinar la imagen de entrada y el mapa de profundidad estimado
    image = cv2.resize(frame, (WIDTH2, HEIGHT2))
    img_out = np.hstack((image, pred_depth_map_colored))

    # Calcular FPS
    fps = fps_counter()  # Calcula los FPS

    # Mostrar FPS en la imagen
    cv2.putText(img_out, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Guardar el frame en el video
    out.write(img_out)

    cv2.imshow("Input and Depth Map", img_out)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar la cámara, cerrar el video y las ventanas
cap.release()
out.release()
cv2.destroyAllWindows()
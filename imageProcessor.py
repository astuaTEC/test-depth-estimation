import cv2
import numpy as np
import tensorflow as tf

# Cargar el modelo TFLite
interpreter = tf.lite.Interpreter(model_path="./models/modelo_preciso.tflite")
interpreter.allocate_tensors()

# Obtener las dimensiones esperadas de entrada para el modelo
input_details = interpreter.get_input_details()[0]
HEIGHT, WIDTH = input_details['shape'][1], input_details['shape'][2]

# Cargar la imagen
image_path = "./imgs/2.jpg"
image = cv2.imread(image_path)
image = cv2.resize(image, (WIDTH, HEIGHT))
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
pred_depth_map_resized = cv2.resize(pred_depth_map.squeeze(), (WIDTH, HEIGHT))

# Convertir el mapa de profundidad redimensionado a una imagen a color para mostrarlo junto a la imagen de entrada
pred_depth_map_colored = cv2.applyColorMap(pred_depth_map_resized, cv2.COLORMAP_MAGMA)

# Combinar la imagen de entrada y el mapa de profundidad estimado
img_out = np.hstack((image, pred_depth_map_colored))

cv2.imshow("Input and Depth Map", img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Guardar el resultado en otra imagen
output_image_path = "./results/resultado.jpg"
cv2.imwrite(output_image_path, img_out)
print("Imagen guardada:", output_image_path)

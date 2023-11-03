# imageProcessor.py
import cv2
import numpy as np
import tensorflow as tf

def imageProcessor(interpreter, input_details, frame_or_image, HEIGHT, WIDTH, HEIGHT2, WIDTH2):

    if isinstance(frame_or_image, str):
        # Si frame_or_image es una cadena, asumimos que es una ruta de imagen
        image = cv2.imread(frame_or_image)
        image = cv2.resize(image, (WIDTH, HEIGHT))
    else:
        # Si frame_or_image es un frame de video, lo usamos directamente
        image = cv2.resize(frame_or_image, (WIDTH, HEIGHT))

    input_data = np.expand_dims(image, axis=0)
    input_data = tf.image.convert_image_dtype(input_data, tf.float32)

    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    pred_depth_map = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

    pred_depth_map = (pred_depth_map - pred_depth_map.min()) / (pred_depth_map.max() - pred_depth_map.min())
    pred_depth_map = (pred_depth_map * 255).astype(np.uint8)

    pred_depth_map_resized = cv2.resize(pred_depth_map.squeeze(), (WIDTH2, HEIGHT2))
    pred_depth_map_resized_inverted = cv2.bitwise_not(pred_depth_map_resized)

    return pred_depth_map_resized_inverted

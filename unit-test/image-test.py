import unittest
import tensorflow as tf
import os
import sys
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Construye la ruta del directorio 'unit-test'
unit_test_dir = os.path.join(SCRIPT_DIR, 'unit-test')

# Agrega 'unit-test' al sys.path si no está presente
if unit_test_dir not in sys.path:
    sys.path.insert(0, unit_test_dir)

from components.imageProcessor import imageProcessor
from components.visualizeResult import visualizeResult

# Carga el modelo TFLite y configura el intérprete
model_name = "unetv8.tflite"
interpreter = tf.lite.Interpreter(model_path=f"./models/{model_name}")
interpreter.allocate_tensors()

# Obtén las dimensiones esperadas de entrada para el modelo
input_details = interpreter.get_input_details()[0]
HEIGHT, WIDTH = input_details['shape'][1], input_details['shape'][2]
HEIGHT2, WIDTH2 = 480, 640

class ImageProcessorTest(unittest.TestCase):

    def test_imageProcessor_saves_output_image(self):
        # Configuración de prueba
        img_source = "./imgs/2.jpg"  # Nombre de archivo de imagen de prueba
        # Asegúrate de que el archivo de imagen de salida se haya creado
        base_name = os.path.splitext(os.path.basename(img_source))[0]
        output_path = f'./unit-test/image/{base_name}_output_depth_map.jpg'
        
        # Llamada a la función que deseas probar
        depth_map = imageProcessor(interpreter, input_details, img_source, HEIGHT, WIDTH, HEIGHT2, WIDTH2)
        
        # Asegúrate de que la salida no sea None (indicando que se procesó la imagen)
        self.assertIsNotNone(depth_map)

        print("\nGuardando...")
        img_out = visualizeResult(cv2.imread(img_source), depth_map, HEIGHT2, WIDTH2)
        
        cv2.imwrite(output_path, img_out)
        print(f"Imagen procesada guardada en {output_path}")

        
        self.assertTrue(os.path.exists(output_path))

if __name__ == '__main__':
    unittest.main()

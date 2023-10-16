import unittest
import tensorflow as tf
import os
import sys
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from components.imageProcessor import imageProcessor
from components.visualizeResult import visualizeResult

class ImageProcessorTest(unittest.TestCase):

    def setUp(self):
        # Cargar el modelo TFLite y configurar el intérprete
        self.model_name = "unetv8.tflite"
        self.interpreter = tf.lite.Interpreter(model_path=f"./models/{self.model_name}")
        self.interpreter.allocate_tensors()

        # Obtener las dimensiones esperadas de entrada para el modelo
        self.input_details = self.interpreter.get_input_details()[0]
        self.HEIGHT, self.WIDTH = self.input_details['shape'][1], self.input_details['shape'][2]
        self.HEIGHT2, self.WIDTH2 = 480, 640

    def test_imageProcessor_saves_output_image(self):
        # Configuración de prueba
        img_source = "./imgs/2.jpg"  # Nombre de archivo de imagen de prueba

        # Verificar que la imagen de entrada está en formato .png o .jpg
        with self.subTest(msg="Verificar formato de la imagen"):
            _, file_extension = os.path.splitext(img_source)
            valid_extensions = ['.png', '.jpg', '.jpeg']
            self.assertIn(file_extension.lower(), valid_extensions)

        base_name = os.path.splitext(os.path.basename(img_source))[0]
        output_path = f'./unit-test/image/{base_name}_output_depth_map.jpg'
        
        # Llamada a la función que deseas probar
        depth_map = imageProcessor(self.interpreter, self.input_details, img_source, self.HEIGHT, self.WIDTH, self.HEIGHT2, self.WIDTH2)
        
        # Asegúrate de que la salida no sea None (indicando que se procesó la imagen)
        self.assertIsNotNone(depth_map)

        print("\nGuardando...")
        img_out = visualizeResult(cv2.imread(img_source), depth_map, self.HEIGHT2, self.WIDTH2)
        
        cv2.imwrite(output_path, img_out)
        print(f"Imagen procesada guardada en {output_path}")

        
        self.assertTrue(os.path.exists(output_path))

if __name__ == '__main__':
    unittest.main()

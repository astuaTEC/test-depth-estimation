import unittest
import tensorflow as tf
import cv2
from components.imageProcessor import imageProcessor

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

    def test_imageProcessor_supports_large_images(self):
        # Configuración de prueba
        large_image_path = "./imgs/3.jpg"  # Ruta de una imagen

        # Verificar que el tamaño de la imagen es mayor a 300x300
        with self.subTest(msg="Verificar tamaño de la imagen"):
            image = cv2.imread(large_image_path)
            height, width, _ = image.shape
            self.assertGreaterEqual(height, 300)
            self.assertGreaterEqual(width, 300)

        # Llamada a la función que deseas probar
        depth_map = imageProcessor(self.interpreter, self.input_details, large_image_path, self.HEIGHT, self.WIDTH, self.HEIGHT2, self.WIDTH2)

        # Asegúrate de que la salida no sea None (indicando que se procesó la imagen)
        self.assertIsNotNone(depth_map)

        # Asegúrate de que el mapa de profundidad tiene las dimensiones correctas
        self.assertEqual(depth_map.shape, (self.HEIGHT2, self.WIDTH2))

if __name__ == '__main__':
    unittest.main()

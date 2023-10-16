import unittest
import tensorflow as tf
import os
import sys
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from components.videoProcessor import videoProcessor

class VideoProcessorTest(unittest.TestCase):

    def setUp(self):
        # Cargar el modelo TFLite y configurar el intérprete
        self.model_name = "unetv8.tflite"
        self.interpreter = tf.lite.Interpreter(model_path=f"./models/{self.model_name}")
        self.interpreter.allocate_tensors()

        # Obtener las dimensiones esperadas de entrada para el modelo
        self.input_details = self.interpreter.get_input_details()[0]
        self.HEIGHT, self.WIDTH = self.input_details['shape'][1], self.input_details['shape'][2]
        self.HEIGHT2, self.WIDTH2 = 480, 640

    def test_videoProcessor_saves_mp4_file(self):

        video_source = "./video-input/test-1.mp4"  # Nombre de archivo de video de prueba

        # Verificar que el video de entrada está en formato MP4
        with self.subTest(msg="Verificar formato del video"):
            _, file_extension = os.path.splitext(video_source)
            self.assertEqual(file_extension.lower(), '.mp4')

        # Llamada a la función que deseas probar
        videoProcessor(self.interpreter, self.input_details, self.HEIGHT, self.WIDTH, self.HEIGHT2, self.WIDTH2, video_source)
        # Asegúrate de que el archivo .mp4 se haya creado
        output_path = f'./unit-test/video/video-output_test-1.mp4'
        self.assertTrue(os.path.exists(output_path))

if __name__ == '__main__':
    unittest.main()
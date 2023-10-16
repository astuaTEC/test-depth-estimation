import unittest
import tensorflow as tf
import os
import sys
import cv2
import time
from unittest.mock import patch

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
        # Llamada a la función que deseas probar con "webcam"
        videoProcessor(self.interpreter, self.input_details, self.HEIGHT, self.WIDTH, self.HEIGHT2, self.WIDTH2, "webcam", wait_time=10)

        # Asegúrate de que el archivo .mp4 se haya creado
        output_path = './unit-test/video/webcam-output.mp4'
        self.assertTrue(os.path.exists(output_path))

if __name__ == '__main__':
    unittest.main()

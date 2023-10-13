import unittest
import tensorflow as tf
import os
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Construye la ruta del directorio 'unit-test'
unit_test_dir = os.path.join(SCRIPT_DIR, 'unit-test')

# Agrega 'unit-test' al sys.path si no está presente
if unit_test_dir not in sys.path:
    sys.path.insert(0, unit_test_dir)

from components.videoProcessor import videoProcessor

# Carga el modelo TFLite y configura el intérprete
model_name = "unetv8.tflite"
interpreter = tf.lite.Interpreter(model_path=f"./models/{model_name}")
interpreter.allocate_tensors()

# Obtén las dimensiones esperadas de entrada para el modelo
input_details = interpreter.get_input_details()[0]
HEIGHT, WIDTH = input_details['shape'][1], input_details['shape'][2]
HEIGHT2, WIDTH2 = 480, 640
class VideoProcessorTest(unittest.TestCase):

    def test_videoProcessor_saves_mp4_file(self):

        video_source = "./video-input/test-1.mp4"  # Nombre de archivo de video de prueba

        # Llamada a la función que deseas probar
        videoProcessor(interpreter, input_details, HEIGHT, WIDTH, HEIGHT2, WIDTH2, video_source, os.path.splitext(model_name)[0])
        # Asegúrate de que el archivo .mp4 se haya creado
        output_path = f'./unit-test/video/video-output_test-1.mp4'
        self.assertTrue(os.path.exists(output_path))

if __name__ == '__main__':
    unittest.main()
import unittest
import tensorflow as tf
import cv2
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

    def test_videoProcessor_supports_large_video_frames(self):
        # Configuración de prueba
        large_video_path = "./video-input/test-1.mp4"  # Nombre de archivo de video de prueba

        # Verificar que el tamaño del primer fotograma del video es mayor a 300x300
        with self.subTest(msg="Verificar tamaño del primer fotograma del video"):
            cap = cv2.VideoCapture(large_video_path)
            _, frame = cap.read()
            cap.release()
            height, width, _ = frame.shape
            self.assertGreaterEqual(height, 300)
            self.assertGreaterEqual(width, 300)

        # Llamada a la función que deseas probar
        videoProcessor(self.interpreter, self.input_details, self.HEIGHT, self.WIDTH, self.HEIGHT2, self.WIDTH2, large_video_path)

        # Aquí se asume que si la función no genera una excepción, consideramos que se ejecutó correctamente.

if __name__ == '__main__':
    unittest.main()

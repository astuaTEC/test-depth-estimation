# main.py
import tensorflow as tf
from components.selector import selectFunctionality
import os

model_name = "unetv5.tflite"

# Cargar el modelo TFLite
interpreter = tf.lite.Interpreter(model_path=f"./models/{model_name}")
interpreter.allocate_tensors()

# Obtener las dimensiones esperadas de entrada para el modelo
input_details = interpreter.get_input_details()[0]
HEIGHT, WIDTH = input_details['shape'][1], input_details['shape'][2]
HEIGHT2, WIDTH2 = 480, 640

def main():
    selectFunctionality(interpreter, input_details, HEIGHT, WIDTH, HEIGHT2, WIDTH2, os.path.splitext(model_name)[0])
    
if __name__ == "__main__":

    main()
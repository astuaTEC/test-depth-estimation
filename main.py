# main.py
import tensorflow as tf
from components.selector import selectFunctionality

# Cargar el modelo TFLite
interpreter = tf.lite.Interpreter(model_path="./models/unetv7.tflite")
interpreter.allocate_tensors()

# Obtener las dimensiones esperadas de entrada para el modelo
input_details = interpreter.get_input_details()[0]
HEIGHT, WIDTH = input_details['shape'][1], input_details['shape'][2]
HEIGHT2, WIDTH2 = 480, 640

csv_handler = CSVHandler('result.csv')

@measure_energy(handler=csv_handler)
def main():
    selectFunctionality(interpreter, input_details, HEIGHT, WIDTH, HEIGHT2, WIDTH2)
    
if __name__ == "__main__":

    main()
    csv_handler.save_data()

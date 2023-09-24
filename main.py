# main.py
import cv2
import tensorflow as tf
from components.videoProcessor import videoProcessor
from components.selector import selectFunctionality
import psutil
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler

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
    # Medir el uso de CPU y memoria antes de llamar a la función crítica
    initial_cpu_percent = psutil.cpu_percent()
    initial_memory_info = psutil.virtual_memory()
    
    print(f"Uso de CPU antes de la función: {initial_cpu_percent}%")
    print(f"Uso de memoria antes de la función: {initial_memory_info.percent}%")

    selectFunctionality(interpreter, input_details, HEIGHT, WIDTH, HEIGHT2, WIDTH2)

    # Medir el uso de CPU y memoria después de llamar a la función crítica
    final_cpu_percent = psutil.cpu_percent()
    final_memory_info = psutil.virtual_memory()
    
    print(f"Uso de CPU después de la función: {final_cpu_percent}%")
    print(f"Uso de memoria después de la función: {final_memory_info.percent}%")
    

if __name__ == "__main__":
    main()
    csv_handler.save_data()

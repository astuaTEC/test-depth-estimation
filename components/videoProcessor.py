import cv2
import numpy as np
import tensorflow as tf
from fps_limiter import FPSCounter
from components.imageProcessor import imageProcessor
from components.visualizeResult import visualizeResult
import time
import psutil
from colorama import Fore, Style

def videoProcessor(interpreter, input_details, HEIGHT, WIDTH, HEIGHT2, WIDTH2, video_source, model_name):
    # Generar un nombre único para el archivo de salida
    current_time = time.strftime("%Y%m%d-%H%M%S")
    # Configuración para guardar el video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if video_source == "webcam":
        cap = cv2.VideoCapture(0)
        output_filename = f'./video/{model_name}/webcam-output_{current_time}.mp4'
    else:
        cap = cv2.VideoCapture(video_source)
        parts = video_source.split('/')
        video_name = parts[-1]
        video_name = video_name.split('.mp4')[0]
        output_filename = f'./video/{model_name}/video-output_{video_name}.mp4'

    out = cv2.VideoWriter(output_filename, fourcc, 10, (2 * WIDTH2, HEIGHT2))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cv2.namedWindow("Input and Depth Map", cv2.WINDOW_NORMAL)

    fps_counter = FPSCounter()

    # Variables acumulativas
    fps_acumulativo = 0
    cpu_acumulativo = 0
    memoria_acumulativa = 0
    num_iteraciones = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Medir el consumo de CPU y memoria
        cpu_percent = psutil.cpu_percent()  # Sin intervalo
        mem_info = psutil.virtual_memory()
        memoria_usada = mem_info.used / (1024 ** 2)  # Convertir a megabytes

        # Imprimir consumo de CPU y memoria
        print(f"\r{Fore.YELLOW}Consumo de CPU:{Style.RESET_ALL} {cpu_percent}%, {Fore.BLUE}Consumo de Memoria:{Style.RESET_ALL} {memoria_usada:.2f} MB", end='', flush=True)

        image = imageProcessor(interpreter, input_details, frame, HEIGHT, WIDTH, HEIGHT2, WIDTH2)
        img_out = visualizeResult(frame, image, HEIGHT2, WIDTH2)

        fps = fps_counter()
        fps_acumulativo += fps
        cpu_acumulativo += cpu_percent
        memoria_acumulativa += memoria_usada
        num_iteraciones += 1

        cv2.putText(img_out, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(img_out)  # Guardar el frame en el video

        cv2.imshow("Input and Depth Map", img_out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Calcular promedios
    fps_promedio = fps_acumulativo / num_iteraciones
    cpu_promedio = cpu_acumulativo / num_iteraciones
    memoria_promedio = memoria_acumulativa / num_iteraciones

    # Imprimir resultados finales
    print(f"\n{Fore.CYAN}Resultados Finales:{Style.RESET_ALL}")
    print(f"{Fore.GREEN}FPS Promedio:{Style.RESET_ALL} {fps_promedio:.4f}")
    print(f"{Fore.MAGENTA}Consumo de CPU Promedio:{Style.RESET_ALL} {cpu_promedio:.4f}%")
    print(f"{Fore.YELLOW}Consumo de Memoria Promedio:{Style.RESET_ALL} {memoria_promedio:.4f} MB")
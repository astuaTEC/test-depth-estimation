import cv2
from components.imageProcessor import imageProcessor
from components.visualizeResult import visualizeResult
import time
from colorama import Fore, Style

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

def videoProcessor(interpreter, input_details, HEIGHT, WIDTH, HEIGHT2, WIDTH2, video_source, wait_time=None):
    # Configuración para guardar el video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if video_source == "webcam":
        cap = cv2.VideoCapture(0)
        output_filename = f'./unit-test/video/webcam-output.mp4'
    else:
        cap = cv2.VideoCapture(video_source)
        parts = video_source.split('/')
        video_name = parts[-1]
        video_name = video_name.split('.mp4')[0]
        output_filename = f'./unit-test/video/video-output_{video_name}.mp4'

    out = cv2.VideoWriter(output_filename, fourcc, 10, (2 * WIDTH2, HEIGHT2))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        print(f"\r{Fore.BLUE}Procesando...{Style.RESET_ALL}", end='', flush=True)


        image = imageProcessor(interpreter, input_details, frame, HEIGHT, WIDTH, HEIGHT2, WIDTH2)
        img_out = visualizeResult(frame, image, HEIGHT2, WIDTH2)

        out.write(img_out)  # Guardar el frame en el video

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if wait_time:
            # Verificar si ha pasado el tiempo deseado
            if time.time() - start_time > wait_time:
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
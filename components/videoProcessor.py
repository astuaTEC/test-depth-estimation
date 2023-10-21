import cv2
from fps_limiter import FPSCounter
from components.imageProcessor import imageProcessor
from components.visualizeResult import visualizeResult
import time

def videoProcessor(interpreter, input_details, HEIGHT, WIDTH, HEIGHT2, WIDTH2, video_source):
    # Generar un nombre único para el archivo de salida
    current_time = time.strftime("%Y%m%d-%H%M%S")
    # Configuración para guardar el video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if video_source == "webcam":
        cap = cv2.VideoCapture(0)
        output_filename = f'./video/webcam-output_{current_time}.mp4'
    else:
        cap = cv2.VideoCapture(video_source)
        parts = video_source.split('/')
        video_name = parts[-1]
        video_name = video_name.split('.mp4')[0]
        output_filename = f'./video/video-output_{video_name}.mp4'

    out = cv2.VideoWriter(output_filename, fourcc, 10, (2 * WIDTH2, HEIGHT2))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cv2.namedWindow("Input and Depth Map", cv2.WINDOW_NORMAL)

    fps_counter = FPSCounter()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        image = imageProcessor(interpreter, input_details, frame, HEIGHT, WIDTH, HEIGHT2, WIDTH2)
        img_out = visualizeResult(frame, image, HEIGHT2, WIDTH2)

        fps = fps_counter()

        cv2.putText(img_out, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(img_out)  # Guardar el frame en el video

        cv2.imshow("Input and Depth Map", img_out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
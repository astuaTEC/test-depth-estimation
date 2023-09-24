# selector.py
import cv2
from components.videoProcessor import videoProcessor
from components.imageProcessor import imageProcessor
from components.visualizeResult import visualizeResult
import numpy as np

def selectFunctionality(interpreter, input_details, HEIGHT, WIDTH, HEIGHT2, WIDTH2):
    print("Selecciona la funcionalidad:")
    print("1. Tomar el video de la cámara web")
    print("2. Tomar un video pregrabado")
    print("3. Aplicar la estimación a una imagen sencilla")
    
    choice = int(input("Ingresa el número de la funcionalidad que deseas: "))
    
    if choice == 1:
        # Tomar el video de la cámara web
        fps = videoProcessor(interpreter, input_details, HEIGHT, WIDTH, HEIGHT2, WIDTH2, "webcam")
        print(f'FPS: {fps:.2f}')
    elif choice == 2:
        # Tomar un video pregrabado
        video_path = "./video-input/test-2.mp4"
        fps = videoProcessor(interpreter, input_details, HEIGHT, WIDTH, HEIGHT2, WIDTH2, video_path)
        print(f'FPS: {fps:.2f}')
    elif choice == 3:
        # Aplicar la estimación a una imagen sencilla
        image_path = "./imgs/2.jpg"
        depth_map = imageProcessor(interpreter, input_details, image_path, HEIGHT, WIDTH, HEIGHT2, WIDTH2)
        if depth_map is not None:
            print("Guardando...")
            img_out = visualizeResult(cv2.imread(image_path), depth_map, HEIGHT2, WIDTH2)
            # Mostrar la imagen
            cv2.imshow("Depth Map", img_out)
            
            # Guardar la imagen
            output_image_path = "./results/output_depth_map.jpg"
            cv2.imwrite(output_image_path, img_out)
            print(f"Imagen procesada guardada en {output_image_path}")
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Opción no válida")

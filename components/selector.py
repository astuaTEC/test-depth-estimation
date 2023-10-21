import os
import cv2
from components.videoProcessor import videoProcessor
from components.imageProcessor import imageProcessor
from components.visualizeResult import visualizeResult
from colorama import Fore, Style

border = f"{Fore.CYAN}{'*' * 30}{Style.RESET_ALL}"  # Línea de borde

def show_functionality_options():
    
    # Imprimir el borde superior
    print(border)
    
    # Imprimir opciones de funcionalidad con formato
    print(f"{Fore.GREEN}1.{Style.RESET_ALL} Tomar el video de la cámara web")
    print(f"{Fore.GREEN}2.{Style.RESET_ALL} Tomar un video pregrabado")
    print(f"{Fore.GREEN}3.{Style.RESET_ALL} Aplicar la estimación a una imagen sencilla")
    
    # Imprimir el borde inferior
    print(border)

def take_webcam_video(interpreter, input_details, HEIGHT, WIDTH, HEIGHT2, WIDTH2):
    videoProcessor(interpreter, input_details, HEIGHT, WIDTH, HEIGHT2, WIDTH2, "webcam")

def list_and_take_video(video_dir, interpreter, input_details, HEIGHT, WIDTH, HEIGHT2, WIDTH2):
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    if not video_files:
        print(f"No hay videos disponibles en el directorio {video_dir}.")
    else:
        print(border)
        print("Videos disponibles:")
        for i, video in enumerate(video_files, start=1):
            print(f"{Fore.GREEN}{i}.{Style.RESET_ALL} {video}")

        print(border)
        video_choice = int(input("Ingresa el número del video que deseas: "))
        if 1 <= video_choice <= len(video_files):
            video_path = os.path.join(video_dir, video_files[video_choice - 1])
            videoProcessor(interpreter, input_details, HEIGHT, WIDTH, HEIGHT2, WIDTH2, video_path)
        else:
            print("Opción no válida")

def list_and_process_image(img_dir, interpreter, input_details, HEIGHT, WIDTH, HEIGHT2, WIDTH2):
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not img_files:
        print(f"No hay imágenes disponibles en el directorio {img_dir}.")
    else:
        print(border)
        print("Imágenes disponibles:")
        for i, img in enumerate(img_files, start=1):
            print(f"{Fore.GREEN}{i}.{Style.RESET_ALL} {img}")

        print(border)
        img_choice = int(input("Ingresa el número de la imagen que deseas: "))
        if 1 <= img_choice <= len(img_files):
            image_path = os.path.join(img_dir, img_files[img_choice - 1])
            depth_map = imageProcessor(interpreter, input_details, image_path, HEIGHT, WIDTH, HEIGHT2, WIDTH2)
            if depth_map is not None:
                base_name = os.path.splitext(img_files[img_choice - 1])[0]  # Nombre sin extensión
                output_image_path = f"./results/{base_name}_output_depth_map.jpg"
                
                print("Guardando...")
                img_out = visualizeResult(cv2.imread(image_path), depth_map, HEIGHT2, WIDTH2)
                cv2.imshow("Depth Map", img_out)

                cv2.imwrite(output_image_path, img_out)
                print(f"Imagen procesada guardada en {output_image_path}")

                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("Opción no válida")

def selectFunctionality(interpreter, input_details, HEIGHT, WIDTH, HEIGHT2, WIDTH2):
    show_functionality_options()
    
    choice = int(input("Ingresa el número de la funcionalidad que deseas: "))
    
    if choice == 1:
        take_webcam_video(interpreter, input_details, HEIGHT, WIDTH, HEIGHT2, WIDTH2)
    elif choice == 2:
        list_and_take_video("./video-input", interpreter, input_details, HEIGHT, WIDTH, HEIGHT2, WIDTH2)
    elif choice == 3:
        list_and_process_image("./imgs", interpreter, input_details, HEIGHT, WIDTH, HEIGHT2, WIDTH2)
    else:
        print("Opción no válida")

# Depth Estimation Script

Este script realiza estimaciones de profundidad en tiempo real a partir de la entrada de una cámara web, videos pregrabados o imágenes.

## Requisitos 🗒️

- Python 3.x
- Instalar dependencias: `pip install -r requirements.txt`

## Uso 💻

1. Clona el repositorio:

    ```bash
    https://github.com/astuaTEC/test-depth-estimation.git
    ```

2. Navega al directorio del script:

    ```bash
    cd test-depth-estimation
    ```

3. Copia tus videos de entrada a la carpeta `video-input`.

4. Copia tus imágenes de entrada a la carpeta `imgs`.

5. Ejecuta el script principal:

    ```bash
    python main.py
    ```

6. Elige la funcionalidad deseada utilizando el selector:

    ```bash
    Selecciona la funcionalidad:
    1. Tomar el video de la cámara web
    2. Tomar un video pregrabado
    3. Aplicar la estimación a una imagen sencilla
    ```

    - Para la opción 1, la estimación se realizará en tiempo real a partir de la cámara web.
    - Para la opción 2, elige el número correspondiente al video pregrabado que deseas procesar.
    - Para la opción 3, elige el número correspondiente a la imagen que deseas procesar.

7. Sigue las instrucciones adicionales según la opción seleccionada.

## Notas 📝

- Los videos de entrada deben ser copiados a la carpeta `video-input`.
- Las imágenes de entrada deben ser copiadas a la carpeta `imgs`.

## Entrenamiento 📚
Para poder entrenar el modelo por cuenta propia, solo siga los siguientes pasos:
1. Cree una cuenta en Kaggle (https://www.kaggle.com/).
2. Cree un proyecto en blanco y luego tiene que subir el archivo `unetv7-kaggle.ipynb` que se encuentra en la raíz del proyecto (`File -> Import Notebook`).
3. En el menú que se encuentra a la derecha del editor de Kaggle, en la parte de `Data`, seleccione `Add data`, busque y seleccione el conjunto de entrenamiento `NYU Depth V2`.
4. En `Notebook options`, seleccione como acelerador la `GPU P100` para que el entrenamiento sea más rapido. Además habilite la opción `Internet on`.
5. En el menú superior seleccione `Run All` y espere a que el entrenamiento termine.

## Créditos 🖋️

Este script fue desarrollado por:

<div align="center"> 
  <br/>
  <b>Saymon Astúa Madrigal - 2018143188</b>
  <br/>
  Instituto Tecnológico de Costa Rica
  <br/><br/>
  <img src="./assets/tec-logo.png" alt="Logo del Instituto Tecnológico de Costa Rica" width="300"/>
</div>



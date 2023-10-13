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

8. Pruebas unitarias

python -m unittest unit-test/video-test.py unit-test/image-test.py

## Notas 📝

- Los videos de entrada deben ser copiados a la carpeta `video-input`.
- Las imágenes de entrada deben ser copiadas a la carpeta `imgs`.

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



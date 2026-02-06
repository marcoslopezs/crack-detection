# Guía de Despliegue en Hugging Face Spaces

Esta guía explica cómo publicar tu proyecto para que cualquiera pueda probarlo desde su navegador.

## 1. ¿Qué archivo necesito?
Para el despliegue público, **SOLO necesitas `streamlit_app.py`** (y los archivos de soporte).
- **`app.py`**: Es tu "backend" (API) para uso local o avanzado. **NO** es necesario para Hugging Face Spaces si usas la versión "All-in-One" (`streamlit_app.py`).
- **`frontend.py`**: Es el cliente local para `app.py`. Tampoco es necesario en la nube.

## 2. Pasos para Desplegar

### Paso 1: Crear el Space
1.  Crea una cuenta en [Hugging Face](https://huggingface.co/join).
2.  Ve a **New Space** (arriba a la derecha).
3.  **Name**: `crack-detection-tfg` (o similar).
4.  **License**: `mit` (opcional).
5.  **SDK**: Selecciona **Streamlit**.
6.  **Hardware**: `CPU Basic (Free)` (Suficiente para este proyecto).
7.  Haz clic en **Create Space**.

### Paso 2: Subir los Archivos
Tienes dos opciones: Web (fácil) o Git (avanzado).

#### Opción A: Subida Web (Recomendada para empezar)
1.  En tu Space, ve a la pestaña **Files**.
2.  Haz clic en **Add file** -> **Upload files**.
3.  Arrastra los siguientes archivos/carpetas de tu proyecto:
    -   `streamlit_app.py`
    -   `model_loader.py`
    -   `model_utils.py`
    -   `requirements.txt`
    -   Carpeta `weights/` (con los .pth dentro)
        -   *Nota: Si los archivos de pesos son muy grandes (>10MB), la web puede tardar. Espera a que suban.*
4.  En "Commit changes", escribe "Initial commit" y pulsa **Commit changes to main**.

### Paso 3: Configuración
Por defecto, Hugging Face busca un archivo `app.py`. Como el tuyo se llama `streamlit_app.py`, debemos indicarlo.

1.  En la pestaña **Files**, edita el archivo `README.md` (el que está en la raíz del Space, no el de tu PC).
2.  Asegúrate de que la cabecera (YAML) tenga esta línea `app_file`:

```yaml
---
title: Crack Detection Tfg
emoji: 🏗️
colorFrom: gray
colorTo: blue
sdk: streamlit
sdk_version: 1.31.0
app_file: streamlit_app.py   <-- AÑADE O CORRIGE ESTA LÍNEA
pinned: false
---
```
3.  Guarda los cambios ("Commit").

### Paso 4: ¡Listo!
Ve a la pestaña **App**. Hugging Face instalará las librerías de `requirements.txt` y arrancará tu aplicación. Tardará unos minutos la primera vez ("Building").

---

## Solución de Problemas Comunes

- **Error de Memoria**: Si se cierra sola, puede que el modelo sea muy grande para la capa gratuita.
    - *Solución*: En `streamlit_app.py`, asegúrate de usar `@st.cache_resource` (ya lo hemos puesto) para no cargar el modelo en cada clic.
- **Error `ImportError: libGL.so.1`**: A veces pasa con OpenCV.
    - *Solución*: Crea un archivo llamado `packages.txt` en el Space y escribe dentro: `libgl1-mesa-glx`.

## Docker (Alternativa)
Si prefieres usar Docker (por ejemplo para desplegar en Render, Railway o AWS), ya tienes el `Dockerfile` listo.
1.  Construir: `docker build -t crack-det .`
2.  Correr: `docker run -p 8501:8501 crack-det`

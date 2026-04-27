# Crack Detection

This repository contains the demo application I built around the crack detection models from my bachelor's thesis. The goal was simple: take the trained models, make them usable from a small web interface, and keep the codebase compact enough to explain in a portfolio.

The project includes two complementary models:

- `Faster R-CNN` for crack localisation with bounding boxes.
- `UNet++` for crack segmentation and basic measurement.

It is not a full inspection platform. It is a focused demo for image-based crack analysis, with the usual caveats around lighting, image quality, camera angle, and domain shift.

## What is in the repo

```text
crack-detection/
|-- app.py                 # FastAPI backend
|-- streamlit_app.py       # Main Streamlit interface
|-- inference_utils.py     # Shared preprocessing and model-loading helpers
|-- model_loader.py        # PyTorch model definitions and weight loading
|-- model_utils.py         # Detection filtering and segmentation metrics
|-- tests/                 # Smoke tests for API and utility functions
|-- weights/               # Trained model weights tracked with Git LFS
|-- requirements.txt       # Exact dependency versions used for validation
|-- .gitattributes         # Git LFS configuration for .pth files
`-- .gitignore
```

## What the app does

- Crack detection mode returns bounding boxes and confidence scores.
- Segmentation mode returns a binary mask overlay plus estimated crack area and length in pixels.
- If you know the image scale, segmentation mode can also convert the measurements to millimetres.

## Local setup

Tested locally with Python `3.12`.

```bash
git clone https://github.com/marcoslopezs/crack-detection.git
cd crack-detection
pip install -r requirements.txt
```

### Run the Streamlit demo

```bash
streamlit run streamlit_app.py
```

### Run the API

```bash
uvicorn app:app --reload
```

Interactive API docs will be available at `http://127.0.0.1:8000/docs`.

## Tests

The repository includes lightweight smoke tests that validate the utility functions and the FastAPI endpoints without requiring full model inference during the test run.

```bash
python -m unittest discover -s tests
```

## Hugging Face Spaces

The project was deployed in Hugging Face Spaces.
It can be seen at https://huggingface.co/spaces/marcoslsantos/crack-detection
## Notes on the models

- The repository ships the trained weights in `weights/`.
- `.pth` files are configured through Git LFS in `.gitattributes`.
- Both the API and the Streamlit UI now use the same preprocessing path, so results are consistent across both entrypoints.

## Limitations

- The segmentation measurements are image-based estimates. They only become physically meaningful when you provide a reliable scale factor.
- Inference on CPU is slower, especially on the first run when the models are loaded.
- The models were trained for a specific crack detection task, so performance can drop on very different surfaces or acquisition conditions.

## License

MIT. See [LICENSE](LICENSE).
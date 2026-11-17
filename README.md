# Lung Cancer Detection

A prototype Streamlit application for exploring lung-image classification. It includes an image-upload interface, a texture-based screening heuristic, and a separate TensorFlow training script.

> **Medical disclaimer:** This project is for educational and experimental use only. It is not a medical device and cannot diagnose or rule out cancer. Its output is not clinically validated or a calibrated probability. Do not use it to make health or treatment decisions; consult a qualified medical professional.

## What the application does

Run `streamlit_app.py` to upload a JPG, JPEG, or PNG image. The app resizes the image to 224 × 224 pixels, calculates the variance of its Laplacian as a texture measure, and compares that value with two hard-coded reference values. It displays the resulting heuristic score and a screening-style label.

**The running app does not load either of the `.h5` model files.** Its score is a simple, unvalidated heuristic, not a model prediction or a diagnostic confidence. The recommendations shown in the interface are generic and are not medical advice.

## Requirements

- Python
- The packages listed in [`requirements.txt`](requirements.txt)

To install the application dependencies and launch it:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit in your browser. The optional page background is loaded from `bgforweb.png`.

## Training

The dataset is expected in class-specific folders:

```text
dataset/
├── cancer/
└── normal/
```

The provided training script uses 224 × 224 images, a 20% validation split, image augmentation, and a frozen ImageNet-pretrained EfficientNetB0 base. It trains for 10 epochs and saves the resulting model as `model.h5` in the project root.

Training requires TensorFlow and Matplotlib, which are **not** included in `requirements.txt`. Install them separately in the environment before running:

```powershell
python -m pip install tensorflow matplotlib
python train.py
```

The training script displays an accuracy plot after training. The model it saves is not connected to `streamlit_app.py`.

## Repository contents

- `streamlit_app.py` — image upload UI and texture-based heuristic
- `train.py` — EfficientNetB0 transfer-learning training script
- `config.py`, `utils.py` — configuration and dataset helper functions
- `check_values.py` — Streamlit utility for displaying image edge density and texture values
- `dataset/` — images organized into `cancer/` and `normal/` directories
- `models/`, `model.h5` — saved model artifacts; not used by the current app
- `bgforweb.png` — optional app background

# ArtMorph AI

> **"Redefine Reality with AI-Powered Artistry"**  
> An end-to-end Neural Style Transfer platform powered by PyTorch and Adaptive Instance Normalization (AdaIN), featuring a custom-trained decoder, multi-hardware acceleration, and dual Flask & Streamlit interfaces.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.2-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Hardware](https://img.shields.io/badge/Device-CUDA%20%7C%20MPS%20%7C%20DirectML%20%7C%20CPU-success?style=flat-square)](https://github.com/microsoft/DirectML)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

---

## 🚀 Live Demo & Links

- **Streamlit Studio (Live Demo)**: [artmorph-ai.streamlit.app](https://artmorph-ai.streamlit.app/) *(Primary demo, fast & cached)*
- **Flask Web App (Render)**: [artmorph-ai.onrender.com](https://artmorph-ai.onrender.com/) *(Runs on free cloud CPU)*
- **GitHub Repository**: [Abhishek18-dev/ArtMorph-AI](https://github.com/Abhishek18-dev/ArtMorph-AI)

---

## 📌 Overview

**ArtMorph AI** is an AI-powered image transformation platform that transfers the artistic style (brushstrokes, colors, and textures) of any reference artwork onto any content photograph in real-time. 

Unlike traditional style transfer algorithms that require hundreds of slow optimization steps per image, ArtMorph AI uses **Adaptive Instance Normalization (AdaIN)** to perform arbitrary style transfer in a **single forward pass**.

This is **not** a third-party API wrapper. The core inversion decoder network was trained from scratch using PyTorch on Google Colab (Tesla T4 GPU). The application is accessible through two distinct web interfaces: an interactive **Streamlit Studio** and a dark cyberpunk-themed **Flask Application**.

---

## ✨ Features

- **Arbitrary Style Transfer**: Apply any art style to any photo in milliseconds without retraining.
- **Custom-Trained PyTorch Decoder**: In-house trained neural network loaded from local checkpoints.
- **Adjustable Style Strength ($\alpha$)**: Smooth slider control from content preservation ($\alpha = 0.0$) to full stylization ($\alpha = 1.0$).
- **Multi-Backend Device Dispatcher**: Auto-detects NVIDIA CUDA, Apple Silicon MPS, Windows AMD/Intel DirectML (`privateuseone:0`), or standard CPU.
- **Dual Web Frontends**:
  - **Streamlit App**: Clean UI with `@st.cache_resource` model caching for instant re-runs and one-click downloads.
  - **Flask App**: Dark sci-fi UI with interactive HTML5 canvas particle background, WTForms validation, and CSRF protection.
- **Wide Format Support**: Handles `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`, and `.gif`.
- **Fault-Tolerant Training**: Training script with checkpoint saving and automatic `--resume` support.

---

## ⚙️ How It Works

```
Content Image + Style Image
         │
         ▼
  VGG-19 Encoder (relu4_1 features)
         │
         ▼
       AdaIN (Aligns feature mean & standard deviation)
         │
         ▼
   Alpha Blending: α · AdaIN + (1 - α) · Content
         │
         ▼
  Trained Decoder (Inverts latent features to RGB)
         │
         ▼
   Stylized Image (Clamped to [0, 1])
```

1. **Feature Extraction**: Content and style images are passed through a frozen, pre-trained VGG-19 encoder to extract deep latent representations at layer `relu4_1`.
2. **Statistical Alignment (AdaIN)**: AdaIN normalizes the content features and shifts them to match the channel-wise mean and variance of the style features, transferring style without losing spatial structure.
3. **Alpha Blending**: The user-selected $\alpha$ value interpolates between raw content features and stylized features.
4. **Decoding**: The trained decoder inverts the blended feature maps back into a viewable RGB image.

---

## 🛠️ Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Language** | Python 3.10 / 3.11 / 3.12 |
| **Deep Learning** | PyTorch 2.4.1, Torchvision 0.19.1 |
| **Model Architecture** | VGG-19 Encoder (frozen) + Custom Inversion Decoder |
| **Core Method** | Adaptive Instance Normalization (AdaIN) |
| **Hardware Acceleration** | CUDA, Apple MPS, DirectML (`torch-directml`), CPU |
| **Web Frameworks** | Streamlit, Flask 3.1.2, Flask-WTF, WTForms, Bootstrap |
| **Image Processing** | Pillow (PIL), NumPy |
| **Production Serving** | Gunicorn (Procfile configuration) |
| **Deployment** | Streamlit Community Cloud, Render |

---

## 📁 Project Structure

```
ArtMorph-AI/
├── NST_AdaIN/
│   ├── static/
│   │   ├── examples/             # Pre-saved content, style, and output pairs
│   │   └── uploads/              # Temporary upload & result storage
│   ├── templates/
│   │   └── index.html            # Dark cyberpunk UI with HTML5 canvas particles
│   ├── utils/
│   │   ├── models.py             # VGGEncoder & Decoder PyTorch definitions
│   │   └── utils.py              # AdaIN calculation, mean/std, dataset loader
│   ├── app.py                    # Flask web application & inference routes
│   ├── streamlit_app.py          # Streamlit app with @st.cache_resource
│   ├── train.py                  # PyTorch training script with resume support
│   ├── commands.txt              # Training execution commands & parameters
│   ├── vgg_normalised.pth        # Pretrained normalized VGG-19 weights (~80 MB)
│   └── decoder_final.pth         # Trained decoder checkpoint (~42 MB)
├── .devcontainer/                # VS Code / Codespaces configuration
├── Procfile                      # Render entrypoint (gunicorn app:app)
├── requirements.txt              # Project dependencies
└── README.md
```

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Abhishek18-dev/ArtMorph-AI.git
cd ArtMorph-AI
```

### 2. Create and Activate Virtual Environment
**Windows (CMD):**
```cmd
python -m venv .venv
.venv\Scripts\activate
```
*(macOS / Linux: `python3 -m venv .venv && source .venv/bin/activate`)*

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Optional for Windows AMD GPU acceleration: `pip install torch-directml`)*

---

## 🏃 Running the Application

### Option A: Streamlit Studio (Recommended)
```bash
cd NST_AdaIN
streamlit run streamlit_app.py
```
Opens in your browser at `http://localhost:8501`. Features cached model loading and fast interactive testing.

### Option B: Flask Cyberpunk Web App
```bash
cd NST_AdaIN
python app.py
```
Opens in your browser at `http://localhost:5000`. Features custom particle animations and glowing UI controls.

---

## 🏋️ Training & Curriculum

The decoder was trained on Google Colab using an **NVIDIA Tesla T4 GPU** across two curriculum stages:

| Stage | Resolution | Batch Size | Epochs | Content Wt ($\lambda_c$) | Style Wt ($\lambda_s$) | Description |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage 1** | $256 \times 256$ | 4 | 160 | 1.0 | 5.0 | Initial convergence from scratch |
| **Stage 2** | $512 \times 512$ | 2 | 200 | 1.0 | 10.0 | Resumed fine-tuning at higher resolution |

- **Training Data**: 49,199 verified images (8,529 MS-COCO content images + 40,670 style collection images; 1 corrupted image filtered out during dataset scan).

### Stage 1 Command (From Scratch)
```bash
cd NST_AdaIN
python train.py --batch_size=4 --epochs=160 --experiment=adain_main --final_size=256 --content_size=512 --style_size=512 --content_weight=1.0 --style_weight=5.0 --content_dir="content_data" --style_dir="style_data" --vgg="vgg_normalised.pth"
```

### Stage 2 Command (Resume & Fine-Tune)
```bash
python train.py --batch_size=2 --epochs=200 --experiment=adain_main --final_size=512 --content_size=512 --style_size=512 --content_weight=1.0 --style_weight=10.0 --content_dir="content_data" --style_dir="style_data" --vgg="vgg_normalised.pth" --resume
```

> **Checkpoint Resumption**: Passing `--resume` automatically scans the experiment folder, loads the checkpoint with the highest epoch, and restores the decoder weights, Adam optimizer states, and learning rate scheduler.

---

## 🧠 Model Summary

- **VGG-19 Encoder**: Sliced up to `relu4_1` (31 layers). Weights are frozen (`requires_grad = False`). Used solely for perceptual feature extraction.
- **AdaIN Layer**: Computes channel-wise statistics $(\mu, \sigma)$ on spatial slices and shifts content feature distributions to match style distributions.
- **Inversion Decoder**: Symmetrically mirrors VGG up to `relu4_1`. Uses reflection padding and nearest-neighbor upsampling ($2\times$) to prevent checkerboard artifacts, ending in a $3\times3$ convolution back to RGB.
- **Style Control ($\alpha$)**: Linearly blends normalized and content features before decoding. $\alpha=1.0$ is full stylization; $\alpha=0.0$ reconstructs the content image.
- **Weight Loading**: Checkpoint `decoder_final.pth` contains a state dictionary dictionary; loaded via `checkpoint['decoder']`.

---

## 🖼️ Results & Examples

### Visual Transformation Flow

| Content Source | Style Reference | Stylized Output ($\alpha = 1.0$) |
| :---: | :---: | :---: |
| `docs/screenshots/content_sample.jpg` | `docs/screenshots/style_sample.jpg` | `docs/screenshots/stylized_sample.jpg` |

*(Pre-saved example pairs are available in [`NST_AdaIN/static/examples/`](file:///c:/Users/Abhishek/ArtMorph-AI/NST_AdaIN/static/examples/))*

---

## ☁️ Deployment

- **Streamlit Community Cloud**: [artmorph-ai.streamlit.app](https://artmorph-ai.streamlit.app/)
  - Primary public demo.
  - Efficient memory retention with `@st.cache_resource`.
- **Render (Flask)**: [artmorph-ai.onrender.com](https://artmorph-ai.onrender.com/)
  - Deployed via Gunicorn and `Procfile`.
  - Runs on free cloud container CPU. Resolution is set to $256 \times 256$ in `app.py` to keep CPU forward-pass latency manageable.
  - *Note: Free cloud instances may take ~50 seconds to wake up from cold sleep.*

---

## ⚠️ Limitations

1. **CPU Latency**: Deep learning inference on CPU-only free cloud tiers takes 4–10 seconds per image compared to <0.3s on GPU.
2. **Ephemeral Storage**: Uploaded and generated images are saved temporarily in `static/uploads/` and clear when cloud containers restart.
3. **High-Resolution VRAM Demands**: Processing images above $1024 \times 1024$ requires substantial GPU memory.
4. **Perceptual Boundaries**: While AdaIN transfers colors and textures reliably, extreme geometric deformation (e.g., transforming a face into a cube) is outside the scope of statistical feature alignment.

---

## 🔮 Future Improvements

- [ ] Add an asynchronous task queue (Celery + Redis) to handle concurrent inference requests.
- [ ] Implement model quantization (ONNX / FP16) to speed up CPU inference.
- [ ] Connect persistent cloud object storage (AWS S3) for generated image galleries.
- [ ] Support multi-style interpolation (blending 2+ style images simultaneously).
- [ ] Add a headless REST API (`/api/v1/stylize`) for external integrations.

---

## 💡 Engineering Highlights

This project demonstrates practical competence in:
- **Deep Learning & Computer Vision**: Neural Style Transfer theory, CNN feature extraction, and AdaIN normalization math.
- **PyTorch Engineering**: Custom `nn.Module` construction, model slicing, multi-loss formulation, and checkpoint serialization.
- **Cross-Platform Acceleration**: Multi-backend dispatcher handling CUDA, Apple Silicon MPS, and Windows AMD DirectML.
- **Full-Stack Development**: Modern Flask application with Jinja2 and Canvas JS alongside a cached Streamlit web app.
- **Production Debugging**: Solving Gunicorn timeouts, WebP format validation, dataset corruptions, and Colab preemption.

---

## 📄 License & Contact

This project is licensed under the **MIT License**.

- **Author**: Abhishek
- **GitHub**: [@Abhishek18-dev](https://github.com/Abhishek18-dev)
- **Repository**: [ArtMorph-AI](https://github.com/Abhishek18-dev/ArtMorph-AI)
- **Live Demo**: [artmorph-ai.streamlit.app](https://artmorph-ai.streamlit.app/)

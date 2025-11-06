# LEDiff: Latent Exposure Diffusion for HDR Generation (CVPR2025)

<p align="center">
  <a href="https://lediff.mpi-inf.mpg.de/">
    <img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page">
  </a>
  <a href="https://lediff.mpi-inf.mpg.de/resource/LEDiff_Latent_Exposure_Diffusion_for_HDR_Generation_Supp.pdf">
    <img src="https://img.shields.io/badge/Paper-PDF-red" alt="Paper">
  </a>
</p>

---

## 📖 <span style="background-color:#f0f0f0; padding:4px 8px; border-radius:6px;">Introduction</span>
LEDiff is a latent diffusion framework for high dynamic range (HDR) image generation.  
The method builds on the [Hugging Face diffusers](https://huggingface.co/docs/diffusers/index) library and adapts exposure-aware latent fusion for robust HDR synthesis.  
It achieves high fidelity in both shadow and highlight regions while remaining efficient for fine-tuning with limited HDR data.

---

## 🚀 <span style="background-color:#f0f0f0; padding:4px 8px; border-radius:6px;">Quick Start</span>

### Installation

```bash
# 1. Install LEDiff package
pip install -e .

# 2. Install example requirements
cd examples/text_to_image
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, CUDA-capable GPU (recommended), and dependencies listed in `examples/text_to_image/requirements.txt` (includes opencv-python, scipy, numpy, torch, ffmpeg-python, etc.)

### Download Pretrained Models

Download the pretrained models from:
- **Highlight Hallucination Model**: [Google Drive](https://drive.google.com/file/d/1gd9KNmOQ3RH4yvX_Fp4hu2Jko64nbt2j/view?usp=sharing)
- **Shadow Hallucination Model**: [Google Drive](https://drive.google.com/file/d/1tMk0rovHSt93wSeIiQ6fxsxPTCiZ6Dmw/view?usp=sharing)

Unzip and note the model folder path.

### Run Inference

**Important:** All paths must be **absolute paths** (starting with `/`), not relative paths.

```bash
cd examples/text_to_image

# Inverse Tone Mapping (ITM)
python test_hdr_itm.py \
  --model_path /home/user/models/lediff_highlight_model/ \
  --image_folder /home/user/LEDiff/dataset/Inverse_Tone_Mapping/ \
  --output_hdr_path /home/user/outputs/ \
  --seed 42

# HDR Generation
python test_hdr_generation.py \
  --model_path /home/user/models/lediff_highlight_model/ \
  --image_folder /home/user/LEDiff/dataset/HDR_generation/ \
  --output_hdr_path /home/user/outputs/
```

**Input formats:**
- ITM: LDR images (`.jpg`, `.png`) in `--image_folder`
- HDR Generation: Latent code `.npy` files in `--image_folder`

**Path requirements:**
- Use full absolute paths (e.g., `/home/user/path/to/folder/`) for all arguments
- Do not use relative paths (e.g., `../folder/` or `./folder/`)
- Ensure paths end with `/` for directories

---

## ⚙️ <span style="background-color:#f0f0f0; padding:4px 8px; border-radius:6px;">Usage for HDR Generation and Reconstruction</span>
For detailed examples and documentation, see: `examples/text_to_image/`

Scripts available:
- **HDR Generation** (`test_hdr_generation.py`)
- **Inverse Tone Mapping** (`test_hdr_itm.py`)



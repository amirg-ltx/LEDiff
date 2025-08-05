# LEDiff: Latent Exposure Diffusion for HDR Generation

<p align="center">
  <a href="YOUR_PROJECT_LINK">
    <img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page">
  </a>
  <a href="YOUR_PAPER_LINK">
    <img src="https://img.shields.io/badge/Paper-PDF-red" alt="Paper">
  </a>
</p>

---

## 📖 <span style="background-color:#f0f0f0; padding:4px 8px; border-radius:6px;">Introduction</span>
LEDiff is a latent diffusion framework for high dynamic range (HDR) image generation.  
The method builds on the [Hugging Face diffusers](https://huggingface.co/docs/diffusers/index) library and adapts exposure-aware latent modeling for robust HDR synthesis.  
It achieves high fidelity in both shadow and highlight regions while remaining efficient for fine-tuning with limited data.

---

## ⚙️ <span style="background-color:#f0f0f0; padding:4px 8px; border-radius:6px;">Environment Setup</span>
Please configure the **diffusers** environment before running LEDiff.  
You can follow the [official diffusers documentation](https://huggingface.co/docs/diffusers/index) or use the minimal setup below.

```bash
# (Optional) Create and activate a clean Python environment
# conda create -n lediff python=3.10 && conda activate lediff

# Install diffusers and related dependencies
pip install --upgrade diffusers transformers accelerate

# Install PyTorch that matches your CUDA or CPU
# See: https://pytorch.org/get-started/locally/
# Example for CUDA 12.x:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

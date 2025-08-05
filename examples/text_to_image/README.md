# HDR Generation and Inverse Tone Mapping Inference

---

## 📖 Introduction
This directory contains:
- **HDR Generation**: Generate high dynamic range (HDR) images from latent codes.
- **Inverse Tone Mapping (ITM)**: Convert low dynamic range (LDR) images to HDR via inverse tone mapping.

---

## 📦 Model Download
Two pretrained **LEDiff** models are available:

- **Highlight Hallucination Model**  
  [Download Link (Google Drive)](https://drive.google.com/file/d/1gd9KNmOQ3RH4yvX_Fp4hu2Jko64nbt2j/view?usp=sharing)

- **Shadow Hallucination Model**  
  [Download Link (Google Drive)](https://drive.google.com/file/d/1tMk0rovHSt93wSeIiQ6fxsxPTCiZ6Dmw/view?usp=sharing)

After downloading, unzip the model files and set `--model_path` to the corresponding local folder.

---

## 📂 Input Data
- **image_folder** should point to the folder containing your input data.  
- For HDR generation, inputs are latent code `.npy` files (e.g., `img_000.npy`, `img_001.npy`).  
- For ITM, inputs are LDR images in `.jpg` / `.png` format.  
- Example input folders in this repository:
  - HDR Generation example:  
    `dataset/HDR_generation/`
  - ITM example:  
    `dataset/Inverse_Tone_Mapping/`



---

## 🚀 Usage
Run HDR Generation and ITM inference:

```bash
# HDR Generation
python test_hdr_generation.py \
  --model_path /path/to/lediff_highlight_model/ \
  --image_folder dataset/HDR_generation/ \
  --output_hdr_path /path/to/save/hdr_results/

# Inverse Tone Mapping
python test_hdr_itm.py \
  --model_path /path/to/lediff_highlight_model/ \
  --image_folder dataset/Inverse_Tone_Mapping/ \
  --output_hdr_path /path/to/save/itm_results/ \
  --seed 42

> The Shadow Hallucination processing is similar to ITM, but requires using the Shadow Hallucination model and changing the pipeline to `StableDiffusionITMUNPipeline`.

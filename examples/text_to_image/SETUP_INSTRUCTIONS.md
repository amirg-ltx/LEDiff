# LEDiff ITM Setup Instructions

This guide explains how to set up and run LEDiff for Inverse Tone Mapping (ITM) on your dataset.

---

## 📋 Prerequisites

Before starting, make sure you have:
- Python 3.8 or higher
- pip (Python package manager)
- CUDA-capable GPU (recommended for faster processing)
- At least 10GB of free disk space for models and outputs

---

## 🚀 Quick Setup (Automated)

### Step 1: Run the Setup Script

Navigate to the text_to_image directory and run the setup script:

```bash
cd /home/user/LEDiff/examples/text_to_image
bash setup_itm.sh
```

**What the script does:**
1. ✅ Installs the LEDiff package (diffusers) in editable mode
2. ✅ Installs all required dependencies from `requirements.txt`
3. ✅ Verifies that all packages are correctly installed

**Expected output:**
- You'll see progress messages for each step
- If successful, you'll see "✅ Setup complete!"
- If there are errors, the script will stop and show what went wrong

---

## 📦 Manual Setup (Alternative)

If you prefer to set up manually or the script doesn't work:

### Step 1: Install LEDiff Package

```bash
cd /home/user/LEDiff
pip install -e .
```

This installs the main LEDiff/diffusers package.

### Step 2: Install Requirements

```bash
cd /home/user/LEDiff/examples/text_to_image
pip install -r requirements.txt
```

This installs all dependencies including:
- `opencv-python` (for image processing)
- `scipy` (for optimization)
- `numpy` (for numerical operations)
- `ffmpeg-python` (for video export support)
- `torch` (PyTorch)
- And other required packages

### Step 3: Verify Installation

Test that everything is installed correctly:

```bash
python -c "import cv2; import numpy; import torch; import scipy; import ffmpeg; from diffusers import StableDiffusionITMPipeline; print('✅ All packages imported successfully')"
```

If you see the success message, you're ready to proceed!

---

## 📥 Download the Pretrained Model

Before running inference, you need to download the pretrained model:

1. **Download the Highlight Hallucination Model** from:
   - [Google Drive Link](https://drive.google.com/file/d/1gd9KNmOQ3RH4yvX_Fp4hu2Jko64nbt2j/view?usp=sharing)

2. **Unzip the downloaded file** to a folder, for example:
   ```bash
   mkdir -p /home/user/models/lediff_highlight
   # Then unzip the model files into this directory
   ```

3. **Note the full path** to the unzipped model folder (you'll need it for the next step)

---

## 🎯 Running ITM Inference

Once setup is complete and you have the model downloaded:

### Basic Command

```bash
cd /home/user/LEDiff/examples/text_to_image

python test_hdr_itm.py \
  --model_path /path/to/lediff_highlight_model/ \
  --image_folder /home/user/LEDiff/dataset/Inverse_Tone_Mapping/ \
  --output_hdr_path /home/user/outputs/ \
  --seed 42
```

### Parameters Explained

- `--model_path`: Full path to the unzipped Highlight Hallucination Model folder
- `--image_folder`: Path to your input LDR images (`.jpg` or `.png` files)
- `--output_hdr_path`: Where to save the generated HDR images
- `--seed`: Random seed for reproducibility (default: 42)

### Example with Your Actual Paths

```bash
python test_hdr_itm.py \
  --model_path /home/user/ComfyUI/models/ltx-ai-labs/lediff/highlight/ \
  --image_folder /home/user/LEDiff/dataset/Inverse_Tone_Mapping/ \
  --output_hdr_path /home/user/outputs/ \
  --seed 42
```

---

## 📝 Important Notes

### Path Format
- **Use absolute paths** (starting with `/`) for reliability
- Make sure paths don't have trailing spaces
- The model path should point to the **folder**, not a specific file

### Command Line Formatting
When using multi-line commands with backslashes (`\`):
- **No spaces after the backslash**
- Each parameter on its own line
- The last line should NOT have a backslash

**Correct:**
```bash
python test_hdr_itm.py \
  --model_path /path/to/model/ \
  --image_folder /path/to/images/ \
  --output_hdr_path /path/to/output/ \
  --seed 42
```

**Incorrect (has trailing space after backslash):**
```bash
python test_hdr_itm.py \ 
  --model_path /path/to/model/
```

### Output Files
The script will create:
- HDR files: `hdr_itm_000.hdr`, `hdr_itm_001.hdr`, etc.
- Intermediate `.npy` files for each processed image

---

## 🔧 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'X'"
**Solution:** Install the missing package:
```bash
pip install X
```

### Error: "CUDA out of memory"
**Solution:** 
- Close other applications using GPU memory
- Process fewer images at a time
- Use a smaller batch size (if the script supports it)

### Error: "Model not found" or "Cannot load model"
**Solution:**
- Verify the `--model_path` points to the correct folder
- Make sure the model files are fully downloaded and unzipped
- Check that the folder contains the required model files (usually `config.json`, `model_index.json`, and weight files)

### Script Runs but No Output
**Solution:**
- Check that `--output_hdr_path` directory exists or can be created
- Verify you have write permissions to the output directory
- Check that input images are in `.jpg`, `.jpeg`, or `.png` format

---

## ✅ Verification Checklist

Before running inference, verify:

- [ ] Setup script completed successfully
- [ ] All packages imported without errors
- [ ] Model downloaded and unzipped
- [ ] Model path is correct and accessible
- [ ] Input image folder contains `.jpg` or `.png` files
- [ ] Output folder path is valid and writable
- [ ] GPU is available (check with `nvidia-smi` if using CUDA)

---

## 📚 Additional Resources

- Main LEDiff README: `/home/user/LEDiff/README.md`
- ITM README: `/home/user/LEDiff/examples/text_to_image/README.md`
- Project Page: https://lediff.mpi-inf.mpg.de/

---

## 🎉 You're Ready!

Once setup is complete, you can run ITM inference on your dataset. The process will convert your LDR images to HDR format using the pretrained LEDiff model.


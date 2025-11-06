#!/bin/bash
# LEDiff ITM (Inverse Tone Mapping) Setup Script
# This script sets up the environment for running LEDiff ITM inference

set -e  # Exit on any error

echo "🚀 LEDiff ITM Setup Script"
echo "=========================="
echo ""

# Get the LEDiff root directory (assuming script is in examples/text_to_image/)
LEDIFF_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
echo "📁 LEDiff root directory: $LEDIFF_ROOT"

# Step 1: Install LEDiff package
echo ""
echo "📦 Step 1: Installing LEDiff package (diffusers)..."
cd "$LEDIFF_ROOT"
pip install -e . || {
    echo "❌ Failed to install LEDiff package"
    exit 1
}
echo "✅ LEDiff package installed"

# Step 2: Install requirements
echo ""
echo "📦 Step 2: Installing ITM requirements..."
cd "$LEDIFF_ROOT/examples/text_to_image"
pip install -r requirements.txt || {
    echo "❌ Failed to install requirements"
    exit 1
}
echo "✅ Requirements installed"

# Step 3: Verify installation
echo ""
echo "🔍 Step 3: Verifying installation..."
python -c "import cv2; import numpy; import torch; import scipy; import ffmpeg; from diffusers import StableDiffusionITMPipeline; print('✅ All packages imported successfully')" || {
    echo "❌ Package verification failed"
    exit 1
}

echo ""
echo "✅ Setup complete! You're ready to run ITM inference."
echo ""
echo "📝 Next steps:"
echo "   1. Download the Highlight Hallucination Model from:"
echo "      https://drive.google.com/file/d/1gd9KNmOQ3RH4yvX_Fp4hu2Jko64nbt2j/view?usp=sharing"
echo "   2. Unzip the model to a folder (e.g., /home/user/models/lediff_highlight/)"
echo "   3. Run the inference script:"
echo "      python test_hdr_itm.py \\"
echo "        --model_path /path/to/lediff_highlight_model/ \\"
echo "        --image_folder /home/user/LEDiff/dataset/Inverse_Tone_Mapping/ \\"
echo "        --output_hdr_path /home/user/outputs/ \\"
echo "        --seed 42"
echo ""


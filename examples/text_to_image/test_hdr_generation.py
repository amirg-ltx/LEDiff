import os
import argparse
import torch
from pathlib import Path
from diffusers import StableDiffusionHDRPipeline
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="HDR generation with LEDiff")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--image_folder", type=str, required=True, help="Folder that contains img_*.npy inputs")
    parser.add_argument("--output_hdr_path", type=str, required=True, help="Output folder for .hdr results")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = args.model_path
    image_folder = args.image_folder
    output_hdr_path = args.output_hdr_path

    # load pipeline
    pipe = StableDiffusionHDRPipeline.from_pretrained(model_path, torch_dtype=torch.float32)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # captions for prompts
    captions = [
        "A glowing candle on a table with a vintage lantern in the background, warm lighting and soft shadows.",
        "A bright, modern living room with large windows, a white sofa, brown curtains, and sunlight streaming onto a shiny wooden floor.",
        "A lit candle in a grand, dimly lit hall with chandeliers and ornate architectural details, viewed from a low angle.",
        "A vibrant city street at night with reflections on a wet surface, colorful lights, and a dreamy, cinematic blur.",
        "A small white candle burning on a wooden pedestal with a neutral background, minimalistic and serene atmosphere.",
        "A modern kitchen with granite countertops, stainless steel appliances, pendant lighting, and a bowl of fruit on the counter.",
        "A tealight candle burning on a wooden surface in a dimly lit room, creating a cozy and intimate ambiance."
    ]

    # collect input files
    image_dir = Path(image_folder)
    image_files = sorted([p for p in image_dir.iterdir() if p.is_file() and p.name.startswith("img_") and p.suffix == ".npy"])

    print("image_files are", [p.name for p in image_files])

    # prepare output dir
    os.makedirs(output_hdr_path, exist_ok=True)

    for idx, img_path in enumerate(image_files):
        str_prompt = captions[idx % len(captions)]
        result = pipe(prompt=str_prompt, npy_name=str(img_path)).images  # expect list like [ldr, hdr_log] or similar
        # HDR saving
        output_hdr_name = os.path.join(output_hdr_path, f"img_{idx:03d}.hdr")
        hdr = np.exp(result[1])  # keep your original logic
        hdr = cv2.cvtColor(hdr.astype(np.float32), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_hdr_name, hdr)


if __name__ == "__main__":
    main()


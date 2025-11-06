import os
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from diffusers import StableDiffusionITMPipeline
try:
    import ffmpeg
except ImportError:
    raise ImportError("Please install ffmpeg-python: pip install ffmpeg-python")
from tqdm import tqdm
import glob
import shutil


def find_video_files(folder_path):
    """
    Find all video files in a folder.
    
    Args:
        folder_path: Path to folder containing videos
    
    Returns:
        List of video file paths
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm']
    video_files = []
    
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    # Search for video files
    for ext in video_extensions:
        video_files.extend(folder.glob(f"*{ext}"))
        video_files.extend(folder.glob(f"*{ext.upper()}"))
    
    # Sort for consistent processing order
    video_files = sorted([str(f) for f in video_files])
    
    return video_files


def extract_frames(video_path, output_folder):
    """Extract all frames from video."""
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Video info: {total_frames} frames at {original_fps} FPS")
    
    frame_paths = []
    frame_count = 0
    
    with tqdm(total=total_frames, desc="    Extracting frames", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_path = os.path.join(output_folder, f"frame_{frame_count:06d}.png")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    return frame_paths, original_fps


def process_frames_to_hdr(pipe, frame_paths, output_folder, prompt, model_path, seed=42):
    """Process each LDR frame to HDR using ITM pipeline."""
    os.makedirs(output_folder, exist_ok=True)
    
    hdr_frame_paths = []
    
    for idx, frame_path in enumerate(tqdm(frame_paths, desc="    Processing to HDR", leave=False)):
        frame_name = Path(frame_path).stem
        npy_save_name = str(Path(output_folder) / f"{frame_name}_latent")
        hdr_output_path = str(Path(output_folder) / f"{frame_name}.hdr")
        
        start_time = time.time()
        
        result = pipe(
            prompt=prompt,
            img_name=frame_path,
            npy_save_name=npy_save_name,
            seed=seed,
            model_path=model_path
        ).images
        
        # Extract HDR image (result[1] is the merged HDR output in log space)
        # The pipeline returns 5 images: [image_tensor, image, image_high, image_medium, image_low]
        hdr_log = result[1]  # This is in log space
        hdr = np.exp(hdr_log)  # Convert from log to linear HDR
        hdr_bgr = cv2.cvtColor(hdr.astype(np.float32), cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(hdr_output_path, hdr_bgr)
        hdr_frame_paths.append(hdr_output_path)
        
        if (idx + 1) % 10 == 0:
            dt = time.time() - start_time
            tqdm.write(f"      Processed {idx+1}/{len(frame_paths)} frames (avg {dt:.2f}s per frame)")
    
    return hdr_frame_paths


def create_exposure_videos_from_exr(exr_folder, videos_output_folder, video_base_name, fps=30, exposure_values=[-3, -2, -1, 0, 1, 2, 3]):
    """
    Create multiple MP4 videos from EXR sequence, each with different exposure values.
    
    Exposure Value (EV) explanation:
    - EV = 0: Neutral exposure (no change)
    - EV < 0: Darker (shows highlights better, e.g., -3 EV = 1/8 brightness)
    - EV > 0: Brighter (shows shadows better, e.g., +3 EV = 8x brightness)
    
    Formula: adjusted_hdr = original_hdr * (2^EV)
    
    Args:
        exr_folder: Path to folder containing EXR files
        videos_output_folder: Folder where output videos will be saved
        video_base_name: Base name for output videos (without extension)
        fps: Frames per second for output videos
        exposure_values: List of exposure values in EV (stops)
    
    Returns:
        List of paths to created video files
    """
    if not os.path.exists(exr_folder):
        raise ValueError(f"EXR folder not found: {exr_folder}")
    
    exr_files = sorted(glob.glob(os.path.join(exr_folder, "*.exr")))
    if not exr_files:
        raise ValueError(f"No EXR files found in {exr_folder}")
    
    first_frame = cv2.imread(exr_files[0], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if first_frame is None:
        raise ValueError(f"Could not read EXR file: {exr_files[0]}")
    
    height, width = first_frame.shape[:2]
    
    os.makedirs(videos_output_folder, exist_ok=True)
    created_videos = []
    
    for ev in exposure_values:
        temp_folder = os.path.join(videos_output_folder, f"temp_ev{ev:+d}")
        os.makedirs(temp_folder, exist_ok=True)
        
        # Calculate exposure multiplier: 2^EV
        # Example: EV=-3 means multiply by 2^-3 = 0.125 (1/8 brightness)
        #          EV=+3 means multiply by 2^3 = 8.0 (8x brightness)
        exposure_multiplier = 2.0 ** ev
        
        for idx, exr_path in enumerate(tqdm(exr_files, desc=f"    EV {ev:+d}", leave=False)):
            # Read EXR file (32-bit float HDR)
            hdr = cv2.imread(exr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            
            if hdr is None:
                continue
            
            # Convert BGR to RGB (OpenCV reads as BGR)
            hdr_rgb = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)
            
            # Apply exposure adjustment
            # Multiply by 2^EV to adjust exposure
            hdr_exposed = hdr_rgb * exposure_multiplier
            
            # Tone map to LDR (8-bit) using Reinhard tone mapping
            # Formula: ldr = (hdr / (hdr + 1.0)) * 255
            # This preserves highlights better than simple gamma
            ldr = np.clip((hdr_exposed / (hdr_exposed + 1.0)) * 255.0, 0, 255).astype(np.uint8)
            
            # Convert back to BGR for saving
            ldr_bgr = cv2.cvtColor(ldr, cv2.COLOR_RGB2BGR)
            
            # Save tone-mapped frame
            frame_path = os.path.join(temp_folder, f"frame_{idx:06d}.png")
            cv2.imwrite(frame_path, ldr_bgr)
        
        # Create video from frames
        output_video_path = os.path.join(videos_output_folder, f"{video_base_name}_ev{ev:+d}.mp4")
        
        try:
            (
                ffmpeg
                .input(f"{temp_folder}/frame_%06d.png", framerate=fps)
                .output(
                    output_video_path,
                    vcodec='libx264',
                    pix_fmt='yuv420p',
                    crf=18,  # High quality
                    preset='medium'  # Encoding speed vs compression
                )
                .overwrite_output()
                .run(quiet=True)
            )
            created_videos.append(output_video_path)
        except Exception as e:
            print(f"      Error encoding EV {ev:+d}: {e}")
        
        # Clean up temp folder
        shutil.rmtree(temp_folder)
    
    return created_videos


def process_single_video(video_path, output_base_folder, pipe, prompt, model_path, seed=42, temp_base_folder="/tmp/lediff_video"):
    """
    Process a single video from LDR to HDR with all exposure outputs.
    
    Args:
        video_path: Path to input video file
        output_base_folder: Base folder where outputs will be organized
        pipe: Loaded ITM pipeline
        prompt: Text prompt for HDR generation
        model_path: Path to model (for pipeline)
        seed: Random seed
        temp_base_folder: Base folder for temporary files
    
    Returns:
        Dictionary with output paths and info
    """
    video_name = Path(video_path).stem  # Get filename without extension
    # Clean video name for folder (remove special characters that might cause issues)
    video_name_clean = "".join(c for c in video_name if c.isalnum() or c in (' ', '-', '_')).strip()
    
    print(f"\n{'='*60}")
    print(f"Processing: {Path(video_path).name}")
    print(f"Video name: {video_name_clean}")
    print(f"{'='*60}")
    
    # Create output folder structure for this video
    video_output_folder = os.path.join(output_base_folder, video_name_clean)
    exr_sequence_folder = os.path.join(video_output_folder, "exr_sequence")
    videos_folder = os.path.join(video_output_folder, "videos")
    
    os.makedirs(exr_sequence_folder, exist_ok=True)
    os.makedirs(videos_folder, exist_ok=True)
    
    # Create temp folders for this video
    video_temp_folder = os.path.join(temp_base_folder, video_name_clean)
    frames_folder = os.path.join(video_temp_folder, "input_frames")
    hdr_frames_folder = os.path.join(video_temp_folder, "hdr_frames")
    
    try:
        # Step 1: Extract frames
        print("\n[Step 1/3] Extracting frames from LDR video...")
        frame_paths, original_fps = extract_frames(video_path, frames_folder)
        
        if not frame_paths:
            raise ValueError("No frames extracted from video")
        
        # Step 2: Process frames to HDR
        print("\n[Step 2/3] Processing frames with ITM pipeline...")
        hdr_frame_paths = process_frames_to_hdr(
            pipe, frame_paths, hdr_frames_folder, prompt, model_path, seed
        )
        
        if not hdr_frame_paths:
            raise ValueError("No HDR frames generated")
        
        # Step 3: Save EXR sequence and create exposure videos
        print("\n[Step 3/3] Creating EXR sequence and exposure videos...")
        
        # Save EXR sequence
        print("  Saving EXR sequence...")
        for idx, hdr_path in enumerate(tqdm(hdr_frame_paths, desc="    Saving EXR", leave=False)):
            hdr = cv2.imread(hdr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if hdr is not None:
                exr_path = os.path.join(exr_sequence_folder, f"frame_{idx:06d}.exr")
                cv2.imwrite(exr_path, hdr)
        
        # Create exposure videos
        print("  Creating exposure videos...")
        exposure_videos = create_exposure_videos_from_exr(
            exr_folder=exr_sequence_folder,
            videos_output_folder=videos_folder,
            video_base_name=video_name_clean,
            fps=original_fps,
            exposure_values=[-3, -2, -1, 0, 1, 2, 3]
        )
        
        print(f"\n✅ Completed: {video_name_clean}")
        print(f"   EXR sequence: {exr_sequence_folder} ({len(hdr_frame_paths)} frames)")
        print(f"   Exposure videos: {len(exposure_videos)} files")
        
        return {
            'video_name': video_name_clean,
            'exr_folder': exr_sequence_folder,
            'videos_folder': videos_folder,
            'exposure_videos': exposure_videos,
            'num_frames': len(hdr_frame_paths),
            'fps': original_fps
        }
    
    except Exception as e:
        print(f"\n❌ Error processing {video_name_clean}: {e}")
        raise
    
    finally:
        # Clean up temp folders (optional - comment out if you want to keep them for debugging)
        if os.path.exists(video_temp_folder):
            shutil.rmtree(video_temp_folder, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Convert LDR videos to HDR videos using LEDiff ITM (batch processing)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing input LDR videos")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder for organized HDR results")
    parser.add_argument("--temp_folder", type=str, default="/tmp/lediff_video", help="Temporary folder for frames")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt for all frames (recommended for consistency)")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load pipeline (only once, reuse for all videos)
    print("\n" + "="*60)
    print("Loading ITM pipeline...")
    print("="*60)
    pipe = StableDiffusionITMPipeline.from_pretrained(args.model_path, torch_dtype=torch.float32)
    pipe.to(device)
    print("✅ Pipeline loaded!")
    
    # Set up prompt
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = "A scene with dynamic lighting and high contrast."
        print(f"\nUsing default prompt: {prompt}")
        print("(Use --prompt to specify a custom prompt)")
    
    # Find all video files
    print("\n" + "="*60)
    print("Scanning for video files...")
    print("="*60)
    video_files = find_video_files(args.input_folder)
    
    if not video_files:
        print(f"❌ No video files found in: {args.input_folder}")
        print("   Supported formats: .mp4, .avi, .mov, .mkv, .flv, .wmv, .m4v, .webm")
        return
    
    print(f"✅ Found {len(video_files)} video(s):")
    for i, vf in enumerate(video_files, 1):
        print(f"   {i}. {Path(vf).name}")
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Process each video
    results = []
    successful = 0
    failed = 0
    
    print("\n" + "="*60)
    print(f"Starting batch processing of {len(video_files)} video(s)...")
    print("="*60)
    
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}] Processing video...")
        
        try:
            result = process_single_video(
                video_path=video_path,
                output_base_folder=args.output_folder,
                pipe=pipe,
                prompt=prompt,
                model_path=args.model_path,
                seed=args.seed,
                temp_base_folder=args.temp_folder
            )
            results.append(result)
            successful += 1
        except Exception as e:
            print(f"❌ Failed to process {Path(video_path).name}: {e}")
            failed += 1
            continue
    
    # Summary
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)
    print(f"✅ Successful: {successful}/{len(video_files)}")
    print(f"❌ Failed: {failed}/{len(video_files)}")
    
    if results:
        print(f"\n📁 Output folder: {args.output_folder}")
        print("\nGenerated structure:")
        for result in results:
            print(f"\n  {result['video_name']}/")
            print(f"    ├── exr_sequence/ ({result['num_frames']} frames)")
            print(f"    └── videos/ ({len(result['exposure_videos'])} exposure videos)")
            for ev_video in result['exposure_videos']:
                print(f"        └── {Path(ev_video).name}")


if __name__ == "__main__":
    main()


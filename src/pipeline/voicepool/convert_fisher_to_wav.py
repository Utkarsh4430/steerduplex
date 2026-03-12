import os
import subprocess
import glob
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Configuration
INPUT_DIR = "/fs/gamma-projects/audio/audio_datasets/fisher"
OUTPUT_DIR = "/fs/gamma-projects/audio/audio_datasets/fisher_wav"
SPH2PIPE_PATH = "/fs/gamma-projects/audio/raman/steerd/steerduplex/voicepool/sph2pipe_v2.5/sph2pipe"

def process_file(args):
    input_path, output_path = args
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Run the sph2pipe command
    command = [SPH2PIPE_PATH, "-f", "wav", input_path, output_path]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing {input_path}: {e}")
        return False

def main():
    # 1. Find all .sph files in directories containing 'sph'
    print(f"Scanning for .sph files in {INPUT_DIR}...")
    sph_files = []
    
    # We look for all subdirectories and just check if the file ends with .sph
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(".sph"):
                sph_files.append(os.path.join(root, file))

    print(f"Found {len(sph_files)} .sph files.")

    # 2. Construct input/output path pairs
    tasks = []
    for input_file in sph_files:
        # Calculate the relative path to maintain directory structure
        rel_path = os.path.relpath(input_file, INPUT_DIR)
        
        # Change extension to .wav
        rel_path_wav = str(Path(rel_path).with_suffix(".wav"))
        
        # Construct full output path
        output_file = os.path.join(OUTPUT_DIR, rel_path_wav)
        
        # Check if output already exists (skip if it does)
        if not os.path.exists(output_file):
             tasks.append((input_file, output_file))

    print(f"Files to process: {len(tasks)}")
    if len(tasks) == 0:
        print("All files processed!")
        return

    # 3. Process files in parallel
    num_cores = os.cpu_count() or 4
    print(f"Using {num_cores} cores for processing.")
    
    success_count = 0
    with Pool(num_cores) as pool:
        # Use imap_unordered for better tqdm progress updates
        for success in tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks), desc="Converting files"):
            if success:
                success_count += 1
                
    print(f"\nCompleted! Successfully converted {success_count}/{len(tasks)} files.")

if __name__ == "__main__":
    main()

import subprocess, os
from pathlib import Path
import statistics
from PIL import Image
from convert import resize, to_jpg, to_raw
from tqdm import tqdm
import json
from itertools import product

NB_RUNS = 5
raw_img_path = Path("Image/base_image.raw")
tmp_img = Path("Image/tmp.png")

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.size

def run(src_path, image_path, block_size, desired_dims=None):
    src_path = Path(src_path)

    if desired_dims is None:
        width, height = get_image_dimensions(image_path)
    else:
        width, height = desired_dims
        image = Image.open(image_path)
        image = resize(image, desired_dims)
        image.save(tmp_img)
        image_path = tmp_img
    
    to_raw(image_path, raw_img_path, (width, height))
    
    with open(src_path/"seuillage.h", "r") as f:
        orig_header_file = f.read()
 
    header_file = orig_header_file.replace("{{SIZE_I}}", str(height))
    header_file = header_file.replace("{{SIZE_J}}", str(width))
    header_file = header_file.replace("{{BLOCK_SIZE}}", str(block_size))
    
    with open(src_path/"seuillage.h", "w") as f:
        f.write(header_file)
    
    old_pwd = os.getcwd()
    os.chdir(src_path/"build")
    if subprocess.run("make", shell=True).returncode != 0:
        raise Exception("Compilation failed")
    
    runs = list()
    for i in tqdm(range(NB_RUNS)):
        runs.append(subprocess.check_output("./seuillage_CUDA ../../Image", shell=True).decode().split("\n"))
    
    results = {
        "cpu_time": [],
        "mem_percentage": [],
        "mem_time": [],
        "mem_size": [],
        "gpu_time": []
    }
    for run in runs:
        for line in run:
            if line.strip() == "": continue
            line = tuple(line.strip().split(" "))
            match line:
                case "memsize:", size, "B":
                    results["mem_size"].append(int(size))
                case "cpu_time:", time, "ms":
                    results["cpu_time"].append(float(time))
                case "gpu_time:", time, "ms":
                    results["gpu_time"].append(float(time))
                case "mem_percentage:", percentage, "percent":
                    results["mem_percentage"].append(float(percentage))
                case "memtime:", time, "ms":
                    results["mem_time"].append(float(time))
                case _:
                    print(line)
        
    os.chdir(old_pwd)
    with open(src_path/"seuillage.h", "w") as f:
        f.write(orig_header_file)
        
    medians = {}
    for key in results:
        medians[key] = statistics.median(results[key])
        
    medians["gpu_compute_time"] = medians["gpu_time"] - medians["mem_time"]
    return medians
        
    
    
if __name__ == "__main__":
    src_paths = ["CUDA int", "CUDA"]
    block_sizes = [2**i for i in range(1, 11)]
    resolutions = [
        (256, 144), (426, 240), (640, 360), (854, 480), (1280, 720),
        (1920, 1080), (2560, 1440), (3840, 2160), (7680, 4320)
    ]

    results = []

    total_combinations = len(src_paths) * len(block_sizes) * len(resolutions)
    for src_path, block_size, dims in tqdm(product(src_paths, block_sizes, resolutions), total=total_combinations):
        result = run(src_path, "Image/Bugatti.jpg", block_size, desired_dims=dims)
        result.update({
            "src_path": src_path,
            "block_size": block_size,
            "resolution": dims
        })
        results.append(result)

        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)
            
        break
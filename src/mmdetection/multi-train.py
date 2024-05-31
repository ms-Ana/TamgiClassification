import os
import subprocess

# Directory containing the configuration files
config_dir = "configs/tamgi"

# List all files in the config directory
config_files = [
    "/home/ana/University/Tamgi/src/mmdetection/configs/tamgi/retinanet_r50_fpn_tamgi.py",
    "/home/ana/University/Tamgi/src/mmdetection/configs/tamgi/yolov3_mobilenetv2_8b24_320_300e_coco_tamgi.py",
]

# Iterate over each configuration file and run the command
for config_file in config_files:
    config_path = os.path.join(config_dir, config_file)
    command = f"python tools/train.py {config_path}"

    try:
        # Execute the command
        subprocess.run(command, shell=True, check=True)
        print(f"Successfully ran: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}\n{e}")

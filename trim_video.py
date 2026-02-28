import argparse
import subprocess
import os

parser = argparse.ArgumentParser(description="Create n clips of 10 seconds each")
parser.add_argument("input_file", help="Input video file")
parser.add_argument("n", type=int, help="Number of clips to create")
args = parser.parse_args()

output_dir = "clips"

# Create clips folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Create n clips of 10 seconds each
for i in range(args.n):
    start_time = i * 10
    output_file = f"{output_dir}/clip_{i}.mp4"

    subprocess.run([
        "ffmpeg", "-y",
        "-i", args.input_file,
        "-ss", str(start_time),
        "-t", "10",
        "-c", "copy",
        output_file
    ])
    print(f"Created {output_file} (seconds {start_time}-{start_time+10})")

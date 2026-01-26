import argparse
import subprocess

parser = argparse.ArgumentParser(description="Trim video to 10 seconds")
parser.add_argument("input_file", help="Input video file")
parser.add_argument("output_file", help="Output video file")
args = parser.parse_args()

subprocess.run([
    "ffmpeg", "-y",
    "-i", args.input_file,
    "-t", "10",
    "-c", "copy",
    args.output_file
])

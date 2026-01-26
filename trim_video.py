import subprocess

input_file = "tracking_overlay.mp4"
output_file = "tracking_overlay_10s.mp4"

subprocess.run([
    "ffmpeg", "-y",
    "-i", input_file,
    "-t", "10",
    "-c", "copy",
    output_file
])

import argparse
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True, help="vLLM server port")
parser.add_argument("--model", type=str, default="/share/j_sun/jqc3/Qwen3.5-27B-FP8", help="Model path")
args = parser.parse_args()

payload = {
    "model": args.model,
    "messages": [
        {"role": "user", "content": "Say hello."}
    ],
    "max_tokens": 10
}

try:
    response = requests.post(
        f"http://localhost:{args.port}/v1/chat/completions",
        json=payload,
        timeout=30
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
except requests.exceptions.Timeout:
    print("Timed out after 30s")
except requests.exceptions.ConnectionError as e:
    print(f"Connection error: {e}")
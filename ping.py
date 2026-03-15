import requests

PORT = 33659

payload = {
    "model": "/share/j_sun/jqc3/Qwen3.5-27B-FP8",
    "messages": [
        {"role": "user", "content": "Say hello."}
    ],
    "max_tokens": 10
}

try:
    response = requests.post(
        f"http://localhost:{PORT}/v1/chat/completions",
        json=payload,
        timeout=30
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
except requests.exceptions.Timeout:
    print("Timed out after 30s")
except requests.exceptions.ConnectionError as e:
    print(f"Connection error: {e}")
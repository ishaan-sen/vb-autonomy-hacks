import requests
import json
import base64

AWS_BEARER_TOKEN = "xxx"

ENDPOINT = "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-lite-v1:0/invoke"

headers = {
    "Authorization": f"Bearer {AWS_BEARER_TOKEN}",
    "Content-Type": "application/json",
}

with open("example.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "text": "Describe this image in detail."
                },
                {
                    "image": {
                        "format": "jpeg",
                        "source": {
                            "bytes": image_b64
                        }
                    }
                }
            ]
        }
    ],
    "inferenceConfig": {
        "maxTokens": 300,
        "temperature": 0.7
    }
}

response = requests.post(ENDPOINT, headers=headers, data=json.dumps(payload))

if response.status_code == 200:
    result = response.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))
else:
    print(f"❌ Failed: {response.status_code} {response.text}")

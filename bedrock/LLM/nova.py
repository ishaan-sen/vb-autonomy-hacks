import requests
import json

AWS_BEARER_TOKEN ="xxx"
# endpoint
ENDPOINT = "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-lite-v1:0/invoke"

headers = {
    "Authorization": f"Bearer {AWS_BEARER_TOKEN}",
    "Content-Type": "application/json",
}

payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"text": "Explain quantum computing in simple terms."}
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

    print("🧠 Answer：")
    print(result["output"]["message"]["content"][0]["text"])

else:
    print(f"❌ Failed: {response.status_code} {response.text}")

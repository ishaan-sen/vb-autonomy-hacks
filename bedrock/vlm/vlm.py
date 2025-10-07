import base64, requests

TOKEN = "xxx"
REGION = "us-east-1"
URL = f"https://bedrock-runtime.{REGION}.amazonaws.com/model/amazon.titan-image-generator-v2:0/invoke"

with open("ex2.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "taskType": "IMAGE_VARIATION",
    "imageVariationParams": {
        "images": [img_b64],          # ✅ 字符串数组
        "text": "cinematic warm sunset lighting, slightly sharper details",
        "similarityStrength": 0.7     # 0.2–1.0（越低越随机）
    },
    "imageGenerationConfig": {        # 推理参数放这里
        "numberOfImages": 1,
        "cfgScale": 8.0,
        "seed": 42,
        # 可选："width":1024,"height":1024（见允许分辨率表）
    }
}

resp = requests.post(
    URL,
    headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json", "Accept": "application/json"},
    json=payload,
)
print(resp.status_code, resp.text[:200])
resp.raise_for_status()
img_b64 = resp.json()["images"][0]
open("variation.png", "wb").write(base64.b64decode(img_b64))
print("✅ 已保存 variation.png")

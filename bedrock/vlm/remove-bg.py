import base64, requests

TOKEN = "xxx"

REGION = "us-east-1"  # 或 us-west-2
URL = f"https://bedrock-runtime.{REGION}.amazonaws.com/model/amazon.titan-image-generator-v2:0/invoke"

# 读入 JPEG 或 PNG，得到“纯 base64 字符串”（不要 data:image/...;base64, 前缀）
with open("ex2.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "taskType": "BACKGROUND_REMOVAL",
    "backgroundRemovalParams": {
        "image": b64  # ✅ 直接传 base64 字符串
    }
}

resp = requests.post(
    URL,
    headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json", "Accept": "application/json"},
    json=payload,
)
print(resp.status_code, resp.text[:200])
resp.raise_for_status()
out = resp.json()
img_b64 = out["images"][0]  # ✅ 返回就是字符串数组
open("no_bg.png", "wb").write(base64.b64decode(img_b64))
print("✅ 已保存 no_bg.png")

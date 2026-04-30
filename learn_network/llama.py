import base64
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# 初始化视觉模型
vision_model = ChatOllama(
    model="llama3.2-vision:11b",          # 使用视觉版模型
    base_url="http://localhost:11434/",
    temperature=0.2
)

# 你的图片路径
image_path = "E://pycharm/program/learn_network/train/school.jpg"


# 将本地图片转为 base64 字符串
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

image_base64 = encode_image(image_path)


# 构造多模态消息：包含文本指令和图片
message = HumanMessage(
    content=[
        {"type": "text", "text": "请详细描述这张图片的内容，包括主要对象、颜色、文字、场景等。"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
        }
    ]
)

# 直接调用模型
response = vision_model.invoke([message])
print("图片描述：", response.content)

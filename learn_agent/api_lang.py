import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


net = ChatOllama(
    model="deepseek-r1:8b",
    base_url="http://localhost:11434/",
    temperature=0.1
)

# message = [
#     SystemMessage(content="你是一个初中数学老师"),
#     HumanMessage(content="老师我在学初中二元一次方程，你能详细解释原理吗？")
# ]
# response = net.invoke(message)
# print(response.content)

# response = net.invoke("你好，我想知道amd的rocm如何进行开发")
# print(response.content)


prompt_template = ChatPromptTemplate([
    ("system", "你是一个乐于助人的助手，请根据用户输入进行回复"),
    ("user", "{user_input}")
])


# prompt = prompt_template.invoke({"user_input":"你能为我解释《明朝儿那些事》这本书吗？"})  # 调用模板
# prompt = prompt_template.format(user_input="你能为我解释《明朝儿那些事》这本书吗？")
# print(prompt)

user_input = "你能为我解释《明朝儿那些事》这本书吗？"
output_parser = StrOutputParser()

chain = prompt_template | net | output_parser

# user_input = "我想知道amd公司开发的ROCM如何进行智能体开发"

for chunk in chain.stream({"user_input": user_input}):
    print(chunk, end="", flush=True)


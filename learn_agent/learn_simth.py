from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


net = ChatOllama(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434/",
    temperature=0.1
)

prompt_template = ChatPromptTemplate([
    ("system", "你是一个乐于助人的助手，请根据用户输入进行回复"),
    ("user", "{user_input}")
])

# print(prompt_template)
# input_variables=['user_input']
# messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[],
#              template='你是一个乐于助人的助手，请根据用户输入进行回复')),
#           HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['user_input'],
#              template='{user_input}'))]

output_parser = StrOutputParser()

# 定义完输入和输出，创建chain
chain = prompt_template | net | output_parser

user_input = "我想知道amd公司开发的ROCM如何进行智能体开发"

for chunk in chain.stream({"user_input": user_input}):
    print(chunk, end="", flush=True)

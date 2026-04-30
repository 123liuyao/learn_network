"""在保留最近对话原始记录的同时，对较早的对话内容进行智能摘要"""
# 以一个客服为例

from langchain.chains.llm import LLMChain
from langchain.memory import ConversationSummaryMemory
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

# 引入本地大模型
net = ChatOllama(
    model='deepseek-r1:8b',
    base_url="http://localhost:11434/",
    temperature=0.2
)
#创建对话模板
prompt = ChatPromptTemplate([
    ("system", "你是电商客服助手，用中文友好回复用户问题。保持专业但亲切的语气"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
#创建带有摘要缓冲的记忆系统
memory = ConversationSummaryMemory(  # 这里如果改成ConversationSummaryBufferMemory以后会有tokenizer的函数需要https调用，所以先改成不用Buffer的版本
    llm=net,
    memory_key='chat_history',
    return_messages=True
)
#创建对话链
chain = LLMChain(
    llm=net,
    memory=memory,
    prompt=prompt,
    verbose=True
)
#创建模拟对话
dialogue = [
    ("你好，我想查询订单12345的状态",None),
    ("这个订单可以退吗",None),
    ("你们的退货政策是什么样的",None),
    ("我不想要了，赶紧为我办理",None)
]

for user_input, _ in dialogue:
    response = chain.invoke({"input": user_input})
    print(f"用户:{user_input}")
    print(f"客服：{response['text']}\n")

print("/n========当前记忆内容=========")
print(memory.load_memory_variables({}))

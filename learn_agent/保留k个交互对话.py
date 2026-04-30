from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain.chains.llm import LLMChain

memory = ConversationBufferWindowMemory(k=3, return_messages=True)

muban = """以下是人类与AI之间的友好对话，AI会比较健谈，可以提供上下午细节。如果AI不知道答案，会表示不知道
当前对话：
{history}
Human:{question}
AI:"""
prompt_template = PromptTemplate(template=muban)


net = ChatOllama(
    model="deepseek-r1:8b",
    base_url="http://localhost:11434/",
    temperature=0.2
)

chain = LLMChain(
    llm=net,
    memory=memory,
    prompt=prompt_template,
    verbose=True
)

respon1 = chain.invoke({"question":"你好，我是孙悟空"})
# print(respon1['text'])
respon2 = chain.invoke({"question":"我还有两个师弟，一个是猪八戒，一个是沙悟净"})
respon3 = chain.invoke({"question":"我今年高考，考上了哈工大"})
respon4 = chain.invoke({"question":"我叫什么?"})
print(respon4['text'])

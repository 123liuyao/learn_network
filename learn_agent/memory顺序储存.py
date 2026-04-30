from langchain.chains.llm import LLMChain
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder

net = ChatOllama(
    model="deepseek-r1:8b",
    base_url="http://localhost:11434/",
    temperature=0.1
)


chat_prompt = ChatPromptTemplate.from_messages([
    ("system","你是一个诗人"),
    MessagesPlaceholder(variable_name='history'),
    ("human", "{question}")
    ])

memory = ConversationBufferMemory(return_messages=True)

chain = LLMChain(prompt=chat_prompt, llm=net, memory=memory)
res1 = chain.invoke({"question":"你好吗，我是唐朝李白"})
print(res1, end='\n\n')

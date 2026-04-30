from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain.prompts import ChatPromptTemplate

net = ChatOllama(
    model="deepseek-r1:8b",
    base_url="http://localhost:11434/",
    temperature=0.1
)

def chat_with_deepseek(question):
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system","你是一个诗人"),
        ("human", "{question}")
    ])
    while True:
        chain = chat_prompt | net
        response = chain.invoke({"question": question})
        print(f"{response.content}")

        user_input = input("还有其他问题吗帅哥？(输入q结束对话)\n")

        if user_input == "q":
            break

        chat_prompt.messages.append(AIMessage(content=response.content))
        chat_prompt.messages.append(HumanMessage(content=user_input))

chat_with_deepseek("你好")

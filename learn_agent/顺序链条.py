from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import SequentialChain,LLMChain
import os
from langchain_ollama import ChatOllama

net = ChatOllama(
    model="deepseek-r1:8b",
    base_url="http://localhost:11434/",
    temperature=0.1
)


chainA_template = ChatPromptTemplate(
    [
        ("system", "你是明朝的诗人于谦"),
        ("human", "请你详细的解释一下：{knowledge}这首诗，并试着{action}")
    ]
)
chainA_chain = LLMChain(
    llm=net,
    prompt=chainA_template,
    verbose=True,
    output_key='chainA_chain_key'
)


chainB_template = ChatPromptTemplate(
    [
        ("system", "你善于提取文本中的重要信息，并做出简短总结"),
        ("human", "这是针对一个提问完整的解释说明内容：{chainA_chain_key}"),
        ("human", "请你根据上述说明，猜一猜第一次提问者在看哪方面的书籍"),
    ]
)
chainB_chain = LLMChain(
    llm=net,
    prompt=chainB_template,
    verbose=True,
    output_key='chainB_chain_key'
)

seq_chain = SequentialChain(
    chains=[chainA_chain, chainB_chain],
    input_variables=["knowledge", "action"],
    output_variables=["chainA_chain_key", "chainB_chain_key"],
    verbose=True
)

response = seq_chain.invoke(
    {
        "knowledge":"《入京》",
        "action":"模仿写一首《出京》"
    }
)
print(response["chainA_chain_key"])  # 单独输出A对话的回答

from langchain.agents import initialize_agent, AgentType,Tool
from langchain_ollama import ChatOllama

net = ChatOllama(
    model='deepseek-r1:8b',
    base_url='http://localhost:11434/',
    temperature=0.2
)

def simple_calculator(expression:str)-> str:
    """基础数学计算工具，支持加减乘除和幂运算
    参数：
        数学表达式，如“3+5”
    返回：
        计算结果字符串或错误信息
    """
    print(f"\n[工具调用]计算表达式：{expression}")
    print("臭男人，怎么才来调用我?")
    return str(eval(expression))

math_calculator_tool = Tool(
    name = 'math_calculator_tool',
    func = simple_calculator,
    description="基础数学计算工具，支持加减乘除和幂运算参数：数学表达式，如“3+5”返回：计算结果字符串或错误信息"
)

agent_exectuor = initialize_agent(
    llm=net,
    agent=AgentType.OPENAI_FUNCTIONS,  # 简单指令模式
    tools=[math_calculator_tool],
    verbose=True,  # 在控制台显示详细得推理过程
    handle_parsing_errors=True
)

response = agent_exectuor.invoke("我想知道2的10次方是多少")
print("最终答案:",response)

# response = net.invoke("我想知道2的10次方是多少")
# print(response)

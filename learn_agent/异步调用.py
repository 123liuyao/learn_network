import asyncio
import time
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# ARK_API_KEY="d3ed22fe-8e85-4e40-a046-235b7f2eef49"

net = ChatOllama(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434/",
    temperature=0.1
)

def sync_test():
    message = [
        SystemMessage(content="你是一个音乐家"),
        HumanMessage(content="请为我介绍汪苏泷的代表作")
    ]
    start = time.time()
    response = net.invoke(message)
    duration = time.time() - start
    print(f"同步调用耗时{duration:.2f}秒")
    return response, duration

async def async_test():
    message1 = [
        SystemMessage(content="你是一个音乐家"),
        HumanMessage(content="请为我介绍汪苏泷的代表作")
    ]
    start = time.time()
    response = await net.ainvoke(message1)
    duration = time.time() - start
    print(f"异步调用耗时{duration:.2f}秒")
    return response, duration

if __name__ == '__main__':
    sync_response, sync_duration = sync_test()
    async_response, async_duration = asyncio.run(async_test())

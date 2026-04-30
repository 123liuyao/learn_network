from langchain_ollama import ChatOllama
from langchain_community.tools import MoveFileTool
from langchain_core.messages import HumanMessage, ToolMessage

# 初始化模型
model = ChatOllama(
    model='gpt-oss:20b',
    base_url="http://localhost:11434/",
    temperature=0.2
)

# 定义工具（这里使用已有的 MoveFileTool，但需要正确配置）
# 注意：MoveFileTool 需要指定源路径和目标路径，你可以根据实际需求创建实例
# 或者使用 @tool 装饰器自定义一个移动文件的工具
# 这里我们直接使用 MoveFileTool 实例并设置描述
move_tool = MoveFileTool()
move_tool.name = "move_file"
move_tool.description = "移动文件到指定位置。输入应为包含 'source' 和 'destination' 键的字典。"

# 绑定工具
model_with_tools = model.bind_tools([move_tool])

# 用户消息
user_message = HumanMessage(content="将文件 E://pycharm/program/learn_network/train/googlenet_train_1.png 移动到 E://pycharm/program/learn_network")

# 第一次调用模型
response = model_with_tools.invoke([user_message])
# print(response.tool_calls)
# 检查是否有工具调用
if response.tool_calls:
    # 执行所有工具调用
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        # 根据名称获取工具实例（这里简化，实际可以用字典映射）
        if tool_name == "move_file":
            # 调用工具
            result = move_tool.invoke(tool_args)
            # 构造工具消息
            tool_message = ToolMessage(
                content=result,  # 工具返回的内容
                tool_call_id=tool_call["id"]  # 必须有 id 匹配
            )
            # 将工具消息追加到历史
            # 如果你希望模型基于工具结果继续回答，可以再次调用模型
            # 这里简单打印结果
            print(f"工具执行结果：{result}")
    # 可选：再次调用模型，让模型基于工具结果生成最终回复
    messages = [user_message, response, tool_message]
    final_response = model_with_tools.invoke(messages)
    print(f"最终回复：{final_response.content}")
else:
    # 没有工具调用，直接输出
    print(response.content)

from langchain_core.messages import (
    BaseMessage,
    SystemMessage, HumanMessage, AIMessage, ToolMessage,
    ToolCall
)
import json
from copy import deepcopy

tuple_msgs = [
    ("system", "你是智能助手"),
    ("human", "7加13等于多少"),
    ("ai", "我需要调用计算工具")
]

obj_msgs1 = [
    SystemMessage(content="你是智能助手"),
    HumanMessage(content="11.4 加 256.5 等于多少"),
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "add",
                "args": {
                    "a": 11.4,
                    "b": 256.5
                },
                "id": "add_0_72ca14f4",
                "type": "tool_call"
            }
        ]
    ),
    ToolMessage(content="267.9", tool_call_id="add_0_72ca14f4"),
    AIMessage(content="11.4 加 256.5 等于 267.9")
]

obj_msgs2 = [
    SystemMessage(content="你是智能助手"),
    HumanMessage(content="11.4 加 256.5 等于多少"),
    AIMessage(
        content="",
        tool_calls=[
            ToolCall( 
                name="add",
                args={
                    "a": 11.4,
                    "b": 256.5
                },
                id="add_0_72ca14f5"
            )
            # 这里的 ToolCall 就是 dict
            # 但它做了强制字段校验
            #   必须包含 name/args/id
            #   可以包含 type (默认值 "tool_call")
        ]
    ),
    ToolMessage(content="267.9", tool_call_id="add_0_72ca14f5"),
    AIMessage(content="11.4 加 256.5 等于 267.9")
]

def show_obj_msgs(obj_msgs:list[BaseMessage]):
    for obj_msg in obj_msgs:
        print(f"[{type(obj_msg).__name__}]", obj_msg.content)
        if isinstance(obj_msg, AIMessage) and obj_msg.tool_calls:
            tool_call0 = obj_msg.tool_calls[0]
            print(f"  [tool call] ({type(tool_call0).__name__})",end=" ")
            if isinstance(tool_call0, dict):
                print(json.dumps(obj_msg.tool_calls, indent=2, ensure_ascii=False))
            elif isinstance(tool_call0, ToolCall):
                # 永远走不到这个分支
                print("    name:", tool_call0.name)
                print("    args:", tool_call0.args)
                print("    id:", tool_call0.id)
        elif isinstance(obj_msg, ToolMessage):
            print("  [tool id]", obj_msg.tool_call_id)
    print("="*50)

# show_obj_msgs(obj_msgs1)
show_obj_msgs(obj_msgs2)

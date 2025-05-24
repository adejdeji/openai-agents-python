import pytest
from openai.types.responses.response_input_item_param import Message

from agents import Agent, Runner, message_tool
from agents.items import ToolMessageItem

from .fake_model import FakeModel
from .test_responses import get_function_tool_call, get_text_message


@message_tool
def simple_msg() -> list[Message]:
    return [{"type": "message", "role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_message_tool_outputs_messages() -> None:
    model = FakeModel()
    agent = Agent(name="test", model=model, tools=[simple_msg])
    model.add_multiple_turn_outputs([
        [get_function_tool_call("simple_msg")],
        [get_text_message("done")],
    ])

    result = await Runner.run(agent, input="hi")

    assert result.final_output == "done"
    # Should include original input, tool call, tool message, and final message
    assert len(result.to_input_list()) == 4
    tool_msg = result.to_input_list()[2]
    assert isinstance(result.new_items[1], ToolMessageItem)
    assert tool_msg["role"] == "user"
    assert tool_msg["content"] == "hello"

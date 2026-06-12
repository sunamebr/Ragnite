from ragnite.llm.base import ChatModel, LLMResponse, Message, SystemPrompt
from ragnite.llm.openai_compat import OpenAICompatChat

__all__ = ["ChatModel", "LLMResponse", "Message", "SystemPrompt", "OpenAICompatChat", "AnthropicChat"]


def __getattr__(name: str):
    if name == "AnthropicChat":  # requires the optional anthropic extra
        from ragnite.llm.anthropic import AnthropicChat

        return AnthropicChat
    raise AttributeError(name)

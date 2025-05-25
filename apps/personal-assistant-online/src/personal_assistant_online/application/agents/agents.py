from pathlib import Path
from typing import Any

import opik
from loguru import logger
from opik import opik_context
from smolagents import LiteLLMModel, MessageRole, MultiStepAgent, ToolCallingAgent

from src.personal_assistant_online.config import settings

def get_agent(retriever_config_path: Path) -> "AgentWrapper":
    agent = AgentWrapper.build_from_smolagents(
        retriever_config_path=retriever_config_path
    )

    return agent


class AgentWrapper:
    def __init__(self, agent: MultiStepAgent) -> None:
        self.__agent = agent

    @property
    def input_messages(self) -> list[dict]:
        return self.__agent.input_messages

    @property
    def agent_name(self) -> str:
        return self.__agent.agent_name

    @property
    def max_steps(self) -> str:
        return self.__agent.max_steps
    
    @classmethod
    def build_from_smolagents(cls, retriever_config_path: Path) -> "AgentWrapper":
        pass

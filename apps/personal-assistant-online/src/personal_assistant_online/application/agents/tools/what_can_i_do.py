import opik
from smolagents import Tool


class WhatCanIDoTool(Tool):
    name = "what_can_i_do"
    description = """Returns a comprehensive list of available capabilities and topics in the Second Brain system.

    This tool should be used when:
    - The user explicitly asks what the system can do
    - The user asks about available features or capabilities
    - The user seems unsure about what questions they can ask
    - The user wants to explore the system's knowledge areas

    This tool should NOT be used when:
    - The user asks a specific technical question
    - The user already knows what they want to learn about
    - The question is about a specific topic covered in the knowledge base
    """

    inputs = {
        "question": {
            "type": "string",
            "description": """The user's query about system capabilities. While this parameter is required,
                             the function returns a standard capability list regardless of the specific question.""",
        }
    }
    output_type = "string"

    @opik.track(name="what_can_i_do")
    def forward(self, question: str) -> str:
        return """
You can ask questions about the content in your Second Brain, such as:

Architecture and Systems:
- What is the feature/training/inference (FTI) architecture?
- How do agentic systems work?
- Detail how does agent memory work in agentic applications?

LLM Technology:
- What are LLMs?
- What is BERT (Bidirectional Encoder Representations from Transformers)?
- Detail how does RLHF (Reinforcement Learning from Human Feedback) work?
- What are the top LLM frameworks for building applications?
- Write me a paragraph on how can I optimize LLMs during inference?

RAG and Document Processing:
- What tools are available for processing PDFs for LLMs and RAG?
- What's the difference between vector databases and vector indices?
- How does document chunk overlap affect RAG performance?
- What is chunk reranking and why is it important?
- What are advanced RAG techniques for optimization?
- How can RAG pipelines be evaluated?

Learning Resources:
- Can you recommend courses on LLMs and RAG?
"""
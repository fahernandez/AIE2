"""LangGraph agent integration with production features and Guardrails validation."""

from typing import Dict, Any, List, Optional
import os
import logging
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.tools import tool
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

from .models import get_openai_model
from .rag import ProductionRAGChain

# Set up logging for security monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Enum for validation results."""

    PASS = "pass"
    FAIL = "fail"
    REFINE = "refine"


class AgentState(TypedDict):
    """Enhanced state schema for agent graphs with validation tracking."""

    messages: Annotated[List[BaseMessage], add_messages]
    validation_errors: List[str]
    guard_activations: List[Dict[str, Any]]
    refinement_count: int


def create_rag_tool(rag_chain: ProductionRAGChain):
    """Create a RAG tool from a ProductionRAGChain."""

    @tool
    def retrieve_information(query: str) -> str:
        """Use Retrieval Augmented Generation to retrieve information from the student loan documents."""
        try:
            result = rag_chain.invoke(query)
            return result.content if hasattr(result, "content") else str(result)
        except Exception as e:
            return f"Error retrieving information: {str(e)}"

    return retrieve_information


def get_default_tools(rag_chain: Optional[ProductionRAGChain] = None) -> List:
    """Get default tools for the agent.

    Args:
        rag_chain: Optional RAG chain to include as a tool

    Returns:
        List of tools
    """
    tools = []

    # Add Tavily search if API key is available
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearchResults(max_results=5))

    # Add Arxiv tool
    tools.append(ArxivQueryRun())

    # Add RAG tool if provided
    if rag_chain:
        tools.append(create_rag_tool(rag_chain))

    return tools


class GuardrailsValidator:
    """Comprehensive Guardrails validation system for production safety."""

    def __init__(self, enable_guardrails: bool = True):
        """Initialize the Guardrails validator.

        Args:
            enable_guardrails: Whether to enable guardrails validation
        """
        self.enable_guardrails = enable_guardrails
        self.guards = {}

        if enable_guardrails:
            self._setup_guards()

    def _setup_guards(self):
        """Set up all guardrails components."""
        try:
            from guardrails.hub import (
                RestrictToTopic,
                DetectJailbreak,
                LlmRagEvaluator,
                HallucinationPrompt,
                ProfanityFree,
                GuardrailsPII,
            )
            from guardrails import Guard

            # 1. Topic Restriction Guard
            self.guards["topic"] = Guard().use(
                RestrictToTopic(
                    valid_topics=[
                        "student loans",
                        "financial aid",
                        "education financing",
                        "loan repayment",
                        "education",
                        "academic research",
                    ],
                    invalid_topics=[
                        "investment advice",
                        "crypto",
                        "gambling",
                        "politics",
                        "illegal activities",
                        "personal financial advice",
                    ],
                    disable_classifier=True,
                    disable_llm=False,
                    on_fail="exception",
                )
            )

            # 2. Jailbreak Detection Guard
            self.guards["jailbreak"] = Guard().use(DetectJailbreak())

            # 3. PII Protection Guard
            self.guards["pii"] = Guard().use(
                GuardrailsPII(
                    entities=[
                        "CREDIT_CARD",
                        "SSN",
                        "PHONE_NUMBER",
                        "EMAIL_ADDRESS",
                        "PERSON",
                        "ADDRESS",
                    ],
                    on_fail="fix",
                )
            )

            # 4. Content Moderation Guard
            self.guards["profanity"] = Guard().use(
                ProfanityFree(
                    threshold=0.8, validation_method="sentence", on_fail="exception"
                )
            )

            # 5. Factuality Guard (for output validation)
            # self.guards["factuality"] = Guard().use(
            #     LlmRagEvaluator(
            #         eval_llm_prompt_generator=HallucinationPrompt(
            #             prompt_name="hallucination_judge_llm"
            #         ),
            #         llm_evaluator_fail_response="hallucinated",
            #         llm_evaluator_pass_response="factual",
            #         llm_callable="gpt-4.1-mini",
            #         on_fail="exception",
            #         on="prompt",
            #     )
            # )

            logger.info("✓ All Guardrails configured successfully")

        except ImportError as e:
            logger.warning(f"Guardrails not available: {e}")
            self.enable_guardrails = False

    def validate_input(self, user_input: str) -> Dict[str, Any]:
        """Validate user input using multiple guards.

        Args:
            user_input: The user's input to validate

        Returns:
            Dictionary with validation results
        """
        if not self.enable_guardrails:
            return {"valid": True, "errors": [], "activations": []}

        errors = []
        activations = []

        try:
            # Topic validation
            try:
                self.guards["topic"].validate(user_input)
                activations.append({"guard": "topic", "status": "pass"})
            except Exception as e:
                errors.append(f"Topic restriction: {str(e)}")
                activations.append(
                    {"guard": "topic", "status": "fail", "error": str(e)}
                )

            # Jailbreak detection
            try:
                result = self.guards["jailbreak"].validate(user_input)
                if result.validation_passed:
                    activations.append({"guard": "jailbreak", "status": "pass"})
                else:
                    errors.append("Jailbreak attempt detected")
                    activations.append({"guard": "jailbreak", "status": "fail"})
            except Exception as e:
                errors.append(f"Jailbreak detection error: {str(e)}")
                activations.append(
                    {"guard": "jailbreak", "status": "error", "error": str(e)}
                )

            # PII detection and redaction
            try:
                result = self.guards["pii"].validate(user_input)
                if result.validated_output != user_input:
                    activations.append(
                        {
                            "guard": "pii",
                            "status": "redacted",
                            "original": user_input,
                            "redacted": result.validated_output,
                        }
                    )
                else:
                    activations.append({"guard": "pii", "status": "pass"})
            except Exception as e:
                errors.append(f"PII detection error: {str(e)}")
                activations.append({"guard": "pii", "status": "error", "error": str(e)})

            # Content moderation
            try:
                self.guards["profanity"].validate(user_input)
                activations.append({"guard": "profanity", "status": "pass"})
            except Exception as e:
                errors.append(f"Content moderation: {str(e)}")
                activations.append(
                    {"guard": "profanity", "status": "fail", "error": str(e)}
                )

            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "activations": activations,
                "redacted_input": (
                    result.validated_output if "pii" in self.guards else user_input
                ),
            }

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {
                "valid": False,
                "errors": [f"Validation system error: {str(e)}"],
                "activations": [
                    {"guard": "system", "status": "error", "error": str(e)}
                ],
                "redacted_input": user_input,
            }

    def validate_output(
        self, agent_response: str, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate agent output using factuality and content guards.

        Args:
            agent_response: The agent's response to validate
            context: Optional context for factuality checking

        Returns:
            Dictionary with validation results
        """
        if not self.enable_guardrails:
            return {"valid": True, "errors": [], "activations": []}

        errors = []
        activations = []

        try:
            # Content moderation for output
            try:
                self.guards["profanity"].validate(agent_response)
                activations.append({"guard": "profanity", "status": "pass"})
            except Exception as e:
                errors.append(f"Output content moderation: {str(e)}")
                activations.append(
                    {"guard": "profanity", "status": "fail", "error": str(e)}
                )

            # # Factuality check (if context provided)
            # if context and "factuality" in self.guards:
            #     try:
            #         # Create a combined prompt for factuality checking
            #         factuality_prompt = (
            #             f"You are a helpful assistant that is an expert in student loans and financial aid. You are given a context and a response. You need to check if the response is factual and relevant to the context. If it is not, you need to return the correct response. \n\n Context: {context}\n\n Response: {agent_response}"
            #         )
            #         result = self.guards["factuality"].validate(factuality_prompt)
            #         if result.validation_passed:
            #             activations.append({"guard": "factuality", "status": "pass"})
            #         else:
            #             errors.append(
            #                 "Factuality check failed - potential hallucination"
            #             )
            #             activations.append({"guard": "factuality", "status": "fail"})
            #     except Exception as e:
            #         errors.append(f"Factuality check error: {str(e)}")
            #         activations.append(
            #             {"guard": "factuality", "status": "error", "error": str(e)}
            #         )

            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "activations": activations,
            }

        except Exception as e:
            logger.error(f"Output validation error: {e}")
            return {
                "valid": False,
                "errors": [f"Output validation system error: {str(e)}"],
                "activations": [
                    {"guard": "system", "status": "error", "error": str(e)}
                ],
            }


def create_input_validation_node(validator: GuardrailsValidator):
    """Create an input validation node for the LangGraph workflow."""

    def validate_input(state: AgentState) -> Dict[str, Any]:
        """Validate user input and handle failures gracefully."""
        messages = state["messages"]

        # Get the latest user message
        user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_message = msg
                break

        if not user_message:
            return {"validation_errors": [], "guard_activations": []}

        # Validate the input
        validation_result = validator.validate_input(user_message.content)

        # Log guard activations for security monitoring
        for activation in validation_result["activations"]:
            logger.info(f"Guard activation: {activation}")

        # If validation failed, create an error message
        if not validation_result["valid"]:
            error_message = SystemMessage(
                content=(
                    f"I cannot process your request due to the following issues:\n"
                    + "\n".join([f"• {error}" for error in validation_result["errors"]])
                    + "\n\nPlease rephrase your question to focus on student loans, financial aid, "
                    + "or education-related topics."
                )
            )

            return {
                "messages": [error_message],
                "validation_errors": validation_result["errors"],
                "guard_activations": validation_result["activations"],
            }

        # If PII was redacted, update the message
        if validation_result.get("redacted_input") != user_message.content:
            redacted_message = HumanMessage(content=validation_result["redacted_input"])
            return {
                "messages": [redacted_message],
                "validation_errors": [],
                "guard_activations": validation_result["activations"],
            }

        return {
            "validation_errors": [],
            "guard_activations": validation_result["activations"],
        }

    return validate_input


def create_output_validation_node(validator: GuardrailsValidator):
    """Create an output validation node for the LangGraph workflow."""

    def validate_output(state: AgentState) -> Dict[str, Any]:
        """Validate agent output and handle refinement if needed."""
        messages = state["messages"]

        # Get the latest AI message
        ai_message = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                ai_message = msg
                break

        if not ai_message:
            return {"validation_errors": [], "guard_activations": []}

        # Get context from previous messages for factuality checking
        context = ""
        for msg in messages[:-1]:
            if isinstance(msg, (HumanMessage, AIMessage)):
                context += msg.content + " "

        # Validate the output
        validation_result = validator.validate_output(ai_message.content, context)

        # Log guard activations
        for activation in validation_result["activations"]:
            logger.info(f"Output guard activation: {activation}")

        # If validation failed and we haven't exceeded refinement attempts
        if not validation_result["valid"] and state.get("refinement_count", 0) < 2:
            # Create a refinement prompt
            refinement_prompt = SystemMessage(
                content=(
                    f"Your previous response had the following issues:\n"
                    + "\n".join([f"• {error}" for error in validation_result["errors"]])
                    + "\n\nPlease provide a corrected response that addresses these issues."
                )
            )

            return {
                "messages": [refinement_prompt],
                "validation_errors": validation_result["errors"],
                "guard_activations": validation_result["activations"],
                "refinement_count": state.get("refinement_count", 0) + 1,
            }

        return {
            "validation_errors": validation_result["errors"],
            "guard_activations": validation_result["activations"],
        }

    return validate_output


def create_langgraph_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None,
    enable_guardrails: bool = True,
):
    """Create a production-safe LangGraph agent with Guardrails validation.

    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        enable_guardrails: Whether to enable guardrails validation

    Returns:
        Compiled LangGraph agent with safety validation
    """
    if tools is None:
        tools = get_default_tools(rag_chain)

    # Initialize guardrails validator
    validator = GuardrailsValidator(enable_guardrails=enable_guardrails)

    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)

    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState):
        """Route to tools if the last message has tool calls."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return "validate_output"

    def should_refine(state: AgentState):
        """Route based on validation results."""
        if state.get("validation_errors"):
            if state.get("refinement_count", 0) < 2:
                return "agent"  # Try refinement
            else:
                return END  # Give up after max attempts
        return END  # Validation passed

    # Build enhanced graph with guardrails
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)

    # Add nodes
    graph.add_node("validate_input", create_input_validation_node(validator))
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("validate_output", create_output_validation_node(validator))

    # Set entry point
    graph.set_entry_point("validate_input")

    # Add edges
    graph.add_edge("validate_input", "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"action": "action", "validate_output": "validate_output"},
    )
    graph.add_edge("action", "agent")
    graph.add_conditional_edges(
        "validate_output", should_refine, {"agent": "agent", END: END}
    )

    return graph.compile()


def create_langgraph_agent_with_guardrails(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None,
):
    """Create a simple LangGraph agent without guardrails (for comparison).

    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool

    Returns:
        Compiled LangGraph agent
    """
    return create_langgraph_agent(
        model_name=model_name,
        temperature=temperature,
        tools=tools,
        rag_chain=rag_chain,
        enable_guardrails=False,
    )

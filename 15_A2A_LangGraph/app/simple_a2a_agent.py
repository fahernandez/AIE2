"""Simple LangGraph agent that uses A2A client for processing user queries."""

import asyncio
import logging
from typing import Dict, Any, Annotated, TypedDict, List
from uuid import uuid4

import httpx
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
)


class AgentState(TypedDict):
    """State schema for the simple A2A agent."""
    messages: Annotated[List, add_messages]
    a2a_response: Any  # Store the A2A response


async def create_a2a_client() -> A2AClient:
    """Create and configure the A2A client."""
    base_url = 'http://localhost:10000'

    # Create HTTP client with increased timeout
    httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))

    # Initialize A2ACardResolver
    resolver = A2ACardResolver(
        httpx_client=httpx_client,
        base_url=base_url,
    )

    # Get the agent card
    try:    
        agent_card = await resolver.get_agent_card()
        return A2AClient(
            httpx_client=httpx_client, 
            agent_card=agent_card
        )
    except Exception as e:
        logging.error(f"Failed to create A2A client: {e}")
        raise


def call_a2a_client(state: Dict[str, Any]) -> Dict[str, Any]:
    """Call the A2A client with the user's query."""
    try:
        # Get the user's message from the state
        messages = state["messages"]
        user_message = None

        # Find the human message
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break

        if not user_message:
            return {
                "messages": [
                    AIMessage(content="Error: No user message found")
                ],
                "a2a_response": None
            }

        # Run the async A2A call in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Create A2A client and send message
            client = loop.run_until_complete(
                create_a2a_client()
            )

            # Prepare the message payload exactly like in test_client.py
            send_message_payload: Dict[str, Any] = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': user_message}
                    ],
                    'message_id': uuid4().hex,
                },
            }

            # Create the request
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )

            # Send the message and get response
            response = loop.run_until_complete(
                client.send_message(request)
            )

            response_dict = response.model_dump(mode='json', exclude_none=True)

            # Extract the response content as the text from the first artifact part
            a2a_content = "No response content"
            try:
                if "result" in response_dict:
                    result = response_dict["result"]
                    # result.artifacts is expected to be a list of dicts with 'parts'
                    artifacts = result.get("artifacts", None)
                    if artifacts and isinstance(artifacts, list):
                        for artifact in artifacts:
                            if artifact.get("parts", None):
                                for part in artifact.get("parts", None):
                                    text = part.get("text", None)
                                    if text:
                                        a2a_content = text
                                        break
            except Exception as e:
                a2a_content = f"Error extracting response content: {e}"

            return {
                "messages": [AIMessage(content=a2a_content)],
                "a2a_response": response
            }

        finally:
            loop.close()

    except Exception as e:
        error_message = f"Error calling A2A client: {str(e)}"
        logging.error(error_message, exc_info=True)
        return {
            "messages": [AIMessage(content=error_message)],
            "a2a_response": None
        }


def build_simple_a2a_graph():
    """Build the simple A2A agent graph with START -> A2A_CALL -> END."""
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("a2a_call", call_a2a_client)
    
    # Set entry point
    workflow.set_entry_point("a2a_call")
    
    # Set exit point
    workflow.add_edge("a2a_call", END)
    
    # Compile the graph
    return workflow.compile()


class SimpleA2AAgent:
    """Simple agent that uses A2A client for processing queries."""
    
    def __init__(self):
        self.graph = build_simple_a2a_graph()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def process_query(self, query: str) -> str:
        """Process a user query using the A2A client."""
        try:
            # Prepare the input
            inputs = {
                "messages": [HumanMessage(content=query)]
            }
            
            # Run the graph
            result = self.graph.invoke(inputs)
            
            # Extract the final response
            final_message = result["messages"][-1]
            if isinstance(final_message, AIMessage):
                return final_message.content
            else:
                return "Error: Unexpected response format"
                
        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            return f"Error processing your request: {str(e)}"


# Example usage
if __name__ == "__main__":
    agent = SimpleA2AAgent()
    
    # Test the agent
    test_query = "What are the latest developments in artificial intelligence?"
    response = agent.process_query(test_query)
    print(f"Query: {test_query}")
    print(f"Response: {response}")

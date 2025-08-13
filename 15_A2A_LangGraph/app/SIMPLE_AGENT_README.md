# Simple A2A LangGraph Agent

This is a simple LangGraph agent that uses the A2A client to process user queries. The agent has a minimal structure with just three nodes:

1. **START** - Entry point
2. **A2A_CALL** - Calls the A2A client with the user's query
3. **END** - Exit point

## Architecture

The agent follows this flow:
```
START → A2A_CALL → END
```

- **START**: Receives the user query and initializes the state
- **A2A_CALL**: Extracts the user message, creates an A2A client, sends the message, and processes the response
- **END**: Returns the final response to the user

## Files

- `simple_a2a_agent.py` - Main agent implementation
- `test_simple_agent.py` - Test script to demonstrate usage

## Usage

### Basic Usage

```python
from simple_a2a_agent import SimpleA2AAgent

# Create the agent
agent = SimpleA2AAgent()

# Process a query
query = "What are the latest developments in artificial intelligence?"
response = agent.process_query(query)
print(response)
```

### Running the Test

```bash
cd app
python test_simple_agent.py
```

### Running the Agent Directly

```bash
cd app
python simple_a2a_agent.py
```

## Requirements

- A2A server running on `http://localhost:10000`
- All dependencies from `pyproject.toml` installed

## How It Works

1. **State Management**: Uses LangGraph's `StateGraph` with a simple state containing messages and A2A response
2. **A2A Integration**: Creates an A2A client using the same pattern as `test_client.py`
3. **Async Handling**: Manages async A2A calls within the synchronous LangGraph node
4. **Response Processing**: Extracts text content from the A2A response and formats it for the user

## Key Features

- **Simple Structure**: Minimal graph with just the essential nodes
- **A2A Protocol**: Uses the A2A client for communication
- **Error Handling**: Graceful error handling with informative messages
- **Logging**: Comprehensive logging for debugging

## Example Output

```
Query: What are the latest developments in artificial intelligence?
Response: [A2A response content will appear here]
```

## Notes

- The agent assumes the A2A server is running on `localhost:10000`
- It uses the same timeout settings as the original test client (60 seconds)
- The agent handles both successful responses and errors gracefully

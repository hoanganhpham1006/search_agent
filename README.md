# Search Agent with Qwen-Agent

A multi-turn search agent built using `qwen_agent` that can search the web and visit URLs to gather information. The agent features reasoning capabilities to improve search queries and synthesize information from multiple sources.

## Features

- **Web Search**: Query web search API with customizable number of results
- **Web Visit**: Retrieve full document content from URLs
- **Multi-turn Conversations**: Up to 5 turns of searching and reasoning
- **Self-improvement**: Agent can refine search queries based on initial results
- **Search History**: Track all searches and visits performed
- **Interactive Mode**: Command-line interface for interactive searching

## Prerequisites

1. **Qwen-Agent**: Install the qwen_agent library
   ```bash
   pip install qwen-agent
   ```

2. **Local Model Server**: Running sglang or vLLM with OpenAI-compatible API at `http://localhost:8000/v1`

3. **Search API Server**: Running at `http://localhost:3000` with endpoints:
   - `POST /search` - Accepts `{query: string, number_results: number}`
   - `POST /visit` - Accepts `{url: string}`

## Installation

1. Clone or download the repository containing:
   - `search_agent.py` - Main agent implementation
   - `config.py` - Configuration file

2. Install dependencies:
   ```bash
   pip install qwen-agent requests
   ```

3. Ensure your model server and search API server are running

## Configuration

All configuration is centralized in `config.py`. Edit this file to customize:

### Model Configuration
```python
MODEL_CONFIG = {
    'model': 'Qwen/Qwen2.5-7B-Instruct',
    'model_server': 'http://localhost:8000/v1',
    'api_key': 'EMPTY',
    'generate_cfg': {
        'fncall_prompt_type': 'qwen'
    },
}
```

### API Endpoints
```python
SEARCH_API_URL = 'http://localhost:3000/search'
VISIT_API_URL = 'http://localhost:3000/visit'
```

### Timeout Settings
```python
SEARCH_TIMEOUT = 30  # seconds
VISIT_TIMEOUT = 60   # seconds
```

### Search Agent Settings
```python
MAX_TURNS = 5              # Maximum conversation turns
DEFAULT_NUM_RESULTS = 1    # Default search results
```

### Custom Configuration
You can still override config at runtime:

```python
from search_agent import SearchAgent

custom_config = {
    'model': 'your-model-name',
    'model_server': 'your-server-url',
    'api_key': 'your-api-key',
}
agent = SearchAgent(model_config=custom_config)
```

## Usage

### Basic Usage

```python
from search_agent import SearchAgent

# Create agent instance
agent = SearchAgent()

# Perform a search
response = agent.search("What are the latest developments in quantum computing?")
print(response)
```

### Interactive Mode

Run the script directly for interactive searching:

```bash
python search_agent.py
```

Commands in interactive mode:
- Type your search query and press Enter
- `history` - View search history
- `reset` - Clear conversation history
- `exit` - Quit the program

### Advanced Usage

```python
# Customize maximum turns
response = agent.search("Complex query requiring multiple searches", max_turns=10)

# Access search history
history = agent.get_search_history()
for item in history:
    print(f"{item['type']}: {item.get('query', item.get('url'))}")

# Reset conversation for new search
agent.reset_conversation()
```

## How It Works

1. **System Prompt**: The agent receives guidelines for effective searching
2. **Function Calling**: Uses qwen_agent's function calling to invoke web_search and web_visit
3. **Reasoning Loop**: 
   - Analyzes user query
   - Performs initial search
   - Reviews results
   - Refines search if needed
   - Visits promising URLs
   - Synthesizes final answer
4. **Response Generation**: Provides comprehensive answer based on gathered information

## API Endpoints

The agent expects these endpoints on your search API server:

### `/search`
```json
Request:
{
    "query": "search terms",
    "number_results": 5
}

Response:
[
    {
        "url": "https://example.com/page1",
        "title": "Page Title"
    },
    ...
]
```

### `/visit`
```json
Request:
{
    "url": "https://example.com/page1"
}

Response:
{
    "content": "Full page content...",
    "url": "https://example.com/page1"
}
```

## Example Output

```
====================================================
User Query: What are the latest developments in quantum computing?
====================================================

--- Turn 1 ---
Calling function: web_search
Arguments: {'query': 'latest developments quantum computing 2024', 'number_results': 3}

--- Turn 2 ---
Calling function: web_visit
Arguments: {'url': 'https://example.com/quantum-breakthrough'}

Agent Response: Based on my search, here are the latest developments in quantum computing...
```

## Project Structure

```
search_agent/
├── search_agent.py    # Main agent implementation
├── config.py          # Configuration settings
└── README.md          # This file
```

## Customizing the System Prompt

Edit `SYSTEM_PROMPT` in `config.py` to change the agent's behavior and capabilities.

## Troubleshooting

1. **Connection Errors**: Ensure both model server and search API are running
2. **Model Not Found**: Verify the model name in `config.py` matches what's loaded in your server
3. **Timeout Issues**: Increase `SEARCH_TIMEOUT` and `VISIT_TIMEOUT` in `config.py`
4. **API Format Issues**: Check that your search API returns the expected JSON format
5. **Import Errors**: Ensure `config.py` is in the same directory as `search_agent.py`

## Benchmarking

The search agent includes a comprehensive benchmarking tool to evaluate performance.

### Running Benchmarks

1. **Prepare your questions dataset**:
   
   CSV format (`questions.csv`):
   ```csv
   question_id,question,category
   q1,What are the latest breakthroughs in quantum computing?,technology
   q2,Compare electric vehicles vs hydrogen fuel cells,environment
   ```
   
   JSON format (`questions.json`):
   ```json
   [
     {"question_id": "q1", "question": "What are the latest breakthroughs in quantum computing?"},
     {"question_id": "q2", "question": "Compare electric vehicles vs hydrogen fuel cells"}
   ]
   ```

2. **Run the benchmark**:
   ```bash
   # Output to console
   python benchmark.py questions.csv
   
   # Save report to file
   python benchmark.py questions.csv -o benchmark_report.txt
   
   # Use the example questions
   python benchmark.py example_questions.csv -o report.txt
   ```

### Benchmark Metrics

The benchmark collects and reports:
- **Response time**: Total time to answer each question
- **Search calls**: Number of web searches performed
- **Visit calls**: Number of URLs visited
- **Success rate**: Percentage of successfully answered questions
- **Full trace**: Complete reasoning steps for each question

### Benchmark Report

The report includes:
1. **Summary statistics**: Overall performance metrics
2. **Results table**: Quick overview of all questions
3. **Detailed results**: For each question:
   - Final answer
   - Execution trace
   - Function call details
   - Error information (if failed)

### Example Benchmark Output

```
================================================================================
SEARCH AGENT BENCHMARK REPORT
================================================================================
Generated: 2024-01-15 10:30:45

SUMMARY
----------------------------------------
Total questions: 5
Successful: 5 (100.0%)
Failed: 0 (0.0%)

Average response time: 8.34s
Average search calls: 2.40
Average visit calls: 1.20

RESULTS TABLE
--------------------------------------------------------------------------------
ID         Time     Search   Visit    Status     Question                      
--------------------------------------------------------------------------------
q1         7.23     2        1        Success    What are the latest breakthr...
q2         9.45     3        2        Success    Compare electric vehicles vs...
```

## License

This implementation follows the Qwen-Agent framework guidelines and is provided as an example implementation.
# Advanced Search Agent with Qwen-Agent

A sophisticated multi-turn search agent built using `qwen_agent` that can search the web and visit URLs to gather information. The agent features advanced reasoning capabilities, transparent thinking processes, and comprehensive benchmarking tools with multi-process support.

## 🚀 Key Features

### **Core Capabilities**
- **Web Search**: Query web search API with customizable number of results
- **Web Visit**: Retrieve full document content from URLs  
- **Multi-turn Conversations**: Unlimited turns with safety mechanisms
- **Transparent Reasoning**: Shows model's thinking process at each step
- **Self-improvement**: Agent refines search queries based on results
- **Search History**: Comprehensive tracking of all searches and visits with timestamps

### **Advanced Features**
- **Mandatory Thinking**: Agent must think in `<think>` tags before every tool call
- **Two-phase Execution**: Planning phase followed by execution phase
- **Reasoning Display**: Extracts and shows thinking from `<think>` tags and `reasoning_content`
- **Function Call Transparency**: Shows why specific tools and arguments were chosen
- **Loop Prevention**: Detects and prevents infinite loops and repeated responses
- **Interactive Mode**: Command-line interface with commands (exit, history, reset)
- **Comprehensive Benchmarking**: Full trace capture with multiprocessing support

### **Benchmark Features**
- **Multiprocessing**: Run benchmarks with configurable number of workers (1-64)
- **Load Balancing**: Round-robin distribution across multiple API ports
- **Full Trace Capture**: Records thinking, reasoning, function calls, and responses
- **Performance Metrics**: Response time, call counts, thinking entries, turn counts
- **Progress Tracking**: Real-time progress updates with ETA
- **Detailed Reports**: Text summaries and JSON traces for analysis

## 📋 Prerequisites

1. **Qwen-Agent**: Install the qwen_agent library
   ```bash
   pip install qwen-agent
   ```

2. **Local Model Server**: Running sglang or vLLM with OpenAI-compatible API at `http://localhost:8000/v1`

3. **Search API Server**: Running with endpoints:
   - `POST /search` - Accepts `{query: string, number_results: number}`
   - `POST /visit` - Accepts `{url: string}`

## 🛠️ Installation

1. Clone or download the repository containing:
   - `search_agent.py` - Main agent implementation
   - `config.py` - Configuration file
   - `benchmark.py` - Benchmarking tool

2. Install dependencies:
   ```bash
   pip install qwen-agent requests pandas
   ```

3. Ensure your model server and search API server are running

## ⚙️ Configuration

All configuration is centralized in `config.py`. Edit this file to customize:

### Model Configuration
```python
MODEL_CONFIG = {
    'model': '/path/to/your/model',
    'model_server': 'http://localhost:8000/v1',
    'api_key': 'EMPTY',
    'generate_cfg': {
        'fncall_prompt_type': 'nous',  # Using nous format for tool calls
        'thought_in_content': True      # Enable thinking in content
    },
}
```

### API Endpoints
```python
SEARCH_API_URL = 'http://192.168.0.14:10000/search'
VISIT_API_URL = 'http://192.168.0.14:10000/visit'
```

### Search Agent Settings
```python
MAX_TURNS = 15             # Maximum conversation turns (safety limit)
DEFAULT_NUM_RESULTS = 3    # Default search results
SEARCH_TIMEOUT = 30        # Search API timeout in seconds
VISIT_TIMEOUT = 60         # Visit API timeout in seconds
```

## 🎯 Usage

### Basic Usage

```python
from search_agent import SearchAgent

# Create agent instance
agent = SearchAgent()

# Perform a search with full reasoning
response = agent.search("What are the latest developments in quantum computing?")
print(response)
```

### Interactive Mode

Run the script directly for interactive searching:

```bash
python search_agent.py
```

**Commands:**
- Type your search query and press Enter
- `history` - View search history
- `reset` - Clear conversation history  
- `exit` - Quit the program

### Advanced Usage

```python
# Access detailed search history
history = agent.get_search_history()
for item in history:
    print(f"{item['type']}: {item.get('query', item.get('url'))}")

# Reset conversation for new search
agent.reset_conversation()

# Custom configuration
custom_config = {
    'model': 'your-model-name',
    'model_server': 'your-server-url',
}
agent = SearchAgent(model_config=custom_config)
```

## 🧠 How It Works

### **Two-Phase Process:**

1. **Planning Phase** (Turn 1):
   - Model thinks about information needs
   - Develops search strategy
   - No tools available - pure reasoning

2. **Execution Phase** (Turn 2+):
   - Performs searches based on plan
   - Visits relevant URLs
   - Shows thinking before each action
   - Synthesizes final comprehensive answer

### **Reasoning Display:**
The agent shows:
- **Model Planning**: Initial strategy and approach
- **Model Thinking**: Reasoning extracted from `<think>` tags
- **Function Calls**: Tool usage with arguments
- **Model Responses**: Analysis and synthesis

### **Safety Mechanisms:**
- **15-turn limit**: Prevents excessive iterations
- **Loop detection**: Stops on repeated responses
- **Final response detection**: Stops when task is complete

## 📊 Example Output

```
============================================================
User Query: What are the latest developments in quantum computing?
============================================================

--- Turn 1 (Planning Phase) ---

Model Planning:
----------------------------------------
I need information about quantum computing, focusing on recent 
developments and breakthroughs. My strategy will be to search 
for authoritative sources and visit the most relevant URLs...
----------------------------------------

--- Turn 2 ---
Calling function: web_search
Arguments: {'query': 'quantum computing explained', 'number_results': 3}

--- Turn 3 ---
Calling function: web_visit
Arguments: {'url': 'https://en.wikipedia.org/wiki/Quantum%20technology'}

--- Turn 4 ---

Model Thinking (from <think> tags):
--------------------------------------------------
The user asked for quantum computing information. I found 
several relevant URLs from my search. The Wikipedia page on 
Quantum Technology seems most comprehensive...
--------------------------------------------------

Model Response:
------------------------------
Quantum computing is a rapidly evolving field that leverages 
quantum mechanics to solve complex problems...
------------------------------

[Model provided final response without function call]

Search History:
- search: quantum computing explained
- visit: https://en.wikipedia.org/wiki/Quantum%20technology
```

## 📈 Benchmarking

The search agent includes a comprehensive benchmarking tool with multiprocessing support that captures complete execution traces.

### **Running Benchmarks**

1. **Prepare Questions Dataset**:
   
   CSV format (`questions.csv`):
   ```csv
   question_id,question
   q1,What are the latest breakthroughs in quantum computing?
   q2,Compare electric vehicles vs hydrogen fuel cells
   ```
   
   JSON format (`questions.json`):
   ```json
   [
     {"question_id": "q1", "question": "What are the latest breakthroughs in quantum computing?"},
     {"question_id": "q2", "question": "Compare electric vehicles vs hydrogen fuel cells"}
   ]
   ```

2. **Run Benchmark**:
   ```bash
   # Basic benchmark with default 48 workers
   python benchmark.py questions.csv
   
   # Multiprocessing with custom worker count
   python benchmark.py questions.csv -w 8 -o benchmark_report.txt
   
   # Save detailed report and traces
   python benchmark.py questions.csv -o benchmark_report.txt
   
   # Disable trace capture (faster, less storage)
   python benchmark.py questions.csv --no-traces -o report.txt
   
   # Verbose output with worker details
   python benchmark.py questions.csv -v
   ```

### **Multiprocessing Configuration**

The benchmark tool includes advanced multiprocessing with:
- **Configurable Workers**: 1-64 workers (default: 48)
- **Load Balancing**: Round-robin across API ports [10000, 11000, 12000, 13000]
- **Progress Tracking**: Real-time updates with ETA and success rates
- **Error Handling**: Graceful failure handling with detailed error reporting

### **Captured Metrics**

The benchmark captures:
- **Response time**: Total execution time
- **Search calls**: Number of web searches performed  
- **Visit calls**: Number of URLs visited
- **Thinking entries**: Count of reasoning steps
- **Total turns**: Number of conversation turns
- **Success rate**: Percentage of completed tasks
- **Full traces**: Complete step-by-step execution

### **Benchmark Output**

**Text Report** (`benchmark_report.txt`):
- Summary statistics
- Performance metrics table
- Detailed results with traces
- Visual trace summaries with emojis

**JSON Report** (`benchmark_report_detailed.json`):
- Complete execution traces
- All thinking and reasoning content
- Function call details
- Timestamps and metadata

### **Trace Format**

Each trace entry includes:
```json
{
  "type": "model_thinking",
  "content": "The user wants information about...",
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

**Trace Types:**
- `user_query`: Original question
- `model_planning`: Initial strategy
- `model_thinking`: Reasoning from `<think>` tags
- `function_call`: Tool usage
- `function_response`: API responses
- `model_response`: Analysis and synthesis
- `final_response`: Complete answer

### **Example Benchmark Report**

```
================================================================================
SEARCH AGENT BENCHMARK REPORT
================================================================================
Generated: 2024-01-15 10:30:45

SUMMARY
----------------------------------------
Total questions: 3
Successful: 3 (100.0%)
Failed: 0 (0.0%)

Performance Metrics:
Average response time: 8.34s
Average search calls: 2.33
Average visit calls: 1.67
Average thinking entries: 4.00
Average turns: 5.33

RESULTS TABLE
----------------------------------------------------------------------------------------------------
ID         Time     Search   Visit    Think    Turns    Status     Question                      
----------------------------------------------------------------------------------------------------
q1         7.23     2        1        3        5        Success    What are the latest breakthr...
q2         9.45     3        2        5        6        Success    Compare electric vehicles vs...

DETAILED RESULTS
====================================================================================================

Question ID: q1
Question: What are the latest breakthroughs in quantum computing?
Status: Success
Response Time: 7.23s
Function Calls: 3 (search: 2, visit: 1)
Thinking Entries: 3
Total Turns: 5

Trace Summary:
--------------------------------------------------
  🧠 Planning: I need information about quantum computing...
  🔧 web_search(query=quantum computing explained, number_results=3)
  📋 Response: 1247 chars
  💭 Thinking: The search results show several relevant URLs...
  🔧 web_visit(url=https://en.wikipedia.org/wiki/Quantum%20technology)
  📋 Response: 5489 chars
  💬 Response: Based on the information gathered...
  ✅ Final: 2847 chars
```

## 🔧 API Endpoints

### `/search`
```json
Request:
{
    "query": "search terms",
    "number_results": 3
}

Response:
{
    "results": [
        {
            "url": "https://example.com/page1",
            "metadata": {"title": "Page Title", "score": 0.95}
        }
    ]
}
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

## 📁 Project Structure

```
search_agent/
├── search_agent.py       # Main agent with reasoning and thinking
├── config.py            # Main configuration settings  
├── benchmark.py         # Advanced benchmarking with multiprocessing
├── benchmark_config.py  # Benchmark-specific configuration
├── example_questions.csv # Sample benchmark data
├── example_output.csv   # Example benchmark output
├── README.md            # This documentation
└── CLAUDE.md            # Detailed project memory and implementation notes
```

## 🎛️ Customization

### System Prompt
Edit `SYSTEM_PROMPT` in `config.py` to modify agent behavior. The current prompt enforces mandatory thinking:

```python
SYSTEM_PROMPT = """You are a helpful search agent with the ability to search the web and visit specific URLs.

🚨 ABSOLUTE RULE: You MUST put your thinking inside <think></think> tags before EVERY tool call. NO EXCEPTIONS!

🚨 MANDATORY FIRST STEP: Always start your response with:
<think>What information does the user need? What's my search strategy? What sources should I prioritize?</think>

🚨 NEVER call a tool without <think></think> tags immediately before it!

CRITICAL WORKFLOW - Follow these steps in order:
1. <think>Think about what information you need and why</think> - then use web_search
2. <think>Evaluate search results: are they relevant or irrelevant? Which URLs should I visit?</think> - then use web_visit
3. <think>What information did I gather? Do I need more?</think> - then either search more or provide answer
"""
```

### Model Configuration
For different thinking modes, models, or APIs:

```python
MODEL_CONFIG = {
    'model': 'your-model-path',
    'model_server': 'your-api-endpoint',
    'generate_cfg': {
        'fncall_prompt_type': 'nous',      # Supports nous/qwen formats
        'thought_in_content': True,        # Enable thinking extraction
        'max_input_tokens': 32000          # Large context for reasoning
    }
}
```

### Benchmark Configuration
The `benchmark_config.py` file contains specialized settings for benchmarking:

```python
# Multiprocessing settings
DEFAULT_NUM_WORKERS = 48
MIN_WORKERS = 1
MAX_WORKERS = 64

# Load balancing across multiple API endpoints
AVAILABLE_PORTS = [10000, 11000, 12000, 13000]

# Comprehensive trace capture
CAPTURE_FULL_TRACES = True
SAVE_DETAILED_JSON = True
```

## 🐛 Troubleshooting

### Common Issues:

1. **No Thinking Displayed**: 
   - Ensure `thought_in_content: True` in model config
   - Verify system prompt enforces `<think>` tags
   - Check if model supports reasoning mode

2. **Function Calls Without Thinking**:
   - Strengthen system prompt with stricter requirements
   - Use mandatory thinking enforcement pattern
   - Check `fncall_prompt_type` setting (nous/qwen)

3. **Infinite Loops**:
   - Verify MAX_TURNS = 15 safety limit
   - Check repeated response detection
   - Monitor turn counter in traces

4. **Connection Errors**: 
   - Ensure model server runs at `localhost:8000/v1`
   - Check search API at `192.168.0.14:10000/search`
   - Verify visit API at `192.168.0.14:10000/visit`
   - Test API connectivity with curl

5. **Benchmark Issues**:
   - Verify questions file CSV/JSON format
   - Check available ports [10000, 11000, 12000, 13000]
   - Ensure worker count within 1-64 range
   - Monitor worker process output for errors

6. **Multiprocessing Issues**:
   - Check system process limits
   - Verify API endpoints can handle concurrent requests
   - Monitor memory usage with many workers
   - Use `-v` flag for verbose worker debugging

## 📝 License

This implementation follows the Qwen-Agent framework guidelines and is provided as an example of advanced agent development with comprehensive reasoning and benchmarking capabilities.
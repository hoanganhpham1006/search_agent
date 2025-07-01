# Benchmark Configuration for Search Agent

# Model Configuration
BENCHMARK_MODEL_CONFIG = {
    'model': '/mnt/sharefs/tuenv/model_hub/qwen3/qwen3-4b',
    'model_server': 'http://localhost:8000/v1',
    'api_key': 'EMPTY',
    'generate_cfg': {
        'fncall_prompt_type': 'nous',  # https://github.com/QwenLM/Qwen-Agent
        'thought_in_content': True,
        'max_input_tokens': 32000,
    },
}

# API Endpoints with load balancing
BENCHMARK_BASE_HOST = '192.168.0.14'
AVAILABLE_PORTS = [10000, 11000, 12000, 13000]

# Timeout Settings (in seconds)
BENCHMARK_SEARCH_TIMEOUT = 30
BENCHMARK_VISIT_TIMEOUT = 60

# Benchmark Process Settings
DEFAULT_NUM_WORKERS = 48
MIN_WORKERS = 1
MAX_WORKERS = 64

# Benchmark Agent Settings
BENCHMARK_MAX_TURNS = 15
BENCHMARK_DEFAULT_NUM_RESULTS = 3

# Benchmark System Prompt (enforces thinking before tool use)
BENCHMARK_SYSTEM_PROMPT = """You are a helpful search agent with the ability to search the web and visit specific URLs.

MANDATORY: You MUST follow this exact pattern for EVERY turn:
1. First, think out loud about your approach
2. Only AFTER explaining your thinking, use tools
3. Continue until you have comprehensive information

CRITICAL RULES:
- NEVER call a function without first explaining your reasoning
- You MUST think before EVERY tool use - no exceptions
- You MUST use web_visit on at least 2-3 URLs from your search results
- Continue searching and visiting URLs until you have comprehensive information

Your capabilities:
- web_search: Find relevant URLs on the web
- web_visit: Read the full content of specific URLs

WORKFLOW:
1. Plan your information gathering strategy
2. Use web_search to find relevant sources
3. Use web_visit to read content from multiple URLs
4. Synthesize information from all sources
5. Provide comprehensive final answer"""

# Trace Capture Settings
CAPTURE_FULL_TRACES = True
SAVE_DETAILED_JSON = True

# Progress Reporting Settings
PROGRESS_UPDATE_FREQUENCY = 10  # Show progress every N% completion
SHOW_WORKER_DETAILS = True
SHOW_ETA = True

# Output Settings
DEFAULT_OUTPUT_SUFFIX = '_detailed.json'

# Safety and Loop Prevention
MAX_REPEATED_RESPONSES = 2
MIN_RESPONSE_LENGTH_FOR_REPEAT_CHECK = 100
SAFETY_DELAY_BETWEEN_QUESTIONS = 0  # seconds

# Performance Metrics
METRICS_TO_TRACK = [
    'response_time',
    'num_search_calls',
    'num_visit_calls',
    'total_function_calls',
    'thinking_entries',
    'total_turns',
    'success_rate'
]

# Error Handling
RETRY_FAILED_REQUESTS = False
LOG_API_ERRORS = True
CONTINUE_ON_WORKER_FAILURE = True

# Port Management
PORT_ROTATION_ENABLED = True
PORT_LOCK_TIMEOUT = 5  # seconds

# Trace Types to Capture
TRACE_TYPES = [
    'user_query',
    'turn_start',
    'model_reasoning',
    'model_thinking',
    'model_response',
    'function_call',
    'function_response',
    'final_response',
    'max_turns_reached',
    'repeated_response_detected'
]

# Report Generation Settings
REPORT_SECTIONS = {
    'summary': True,
    'performance_metrics': True,
    'results_table': True,
    'detailed_results': True,
    'full_conversation_traces': True
}


# Worker Process Settings
WORKER_SETTINGS = {
    'start_method': 'spawn',  # or 'fork' depending on system
    'timeout': 300,  # seconds per question
    'memory_limit': None,  # MB, None for no limit
}

# Validation Settings
VALIDATE_PORTS_ON_STARTUP = False
PING_TIMEOUT = 5  # seconds for port validation

# Debug Settings
DEBUG_MODE = False
VERBOSE_WORKER_OUTPUT = False
LOG_INCOMPLETE_TOOL_CALLS = True
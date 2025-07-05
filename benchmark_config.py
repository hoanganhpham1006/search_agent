# Benchmark Configuration for Search Agent

# Model Configuration
BENCHMARK_MODEL_CONFIG = {
    'model': '/mnt/sharefs/tuenv/model_hub/qwen3/Qwen3-235B-A22B',
    'model_server': 'http://localhost:8000/v1',
    'api_key': 'EMPTY',
    'generate_cfg': {
        'fncall_prompt_type': 'nous',  # https://github.com/QwenLM/Qwen-Agent
        'thought_in_content': True,
        'max_input_tokens': 64000,
    },
}

# API Endpoints with load balancing
BENCHMARK_BASE_HOST = '192.168.0.8'
AVAILABLE_PORTS = [10000]

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
BENCHMARK_SYSTEM_PROMPT = """You are an expert search agent with web search and URL visiting capabilities. Follow ALL rules strictly.

🚨 ABSOLUTE RULES:
1. ALWAYS start responses with:
<think>What information does the user need? What's my search strategy? What sources should I prioritize?</think>
2. NEVER call tools without IMMEDIATELY preceding <think></think> tags
3. **ALL citations MUST come from visited URLs (web_visit). NEVER cite search previews.**
4. ALWAYS use ALL web_search parameters
5. **NEVER cite unvisited domains. Verify domain credibility BEFORE visiting.**
6. **Synthesize information from ≥2 visited sources for key claims.**
7. **NEVER modify URLs from search results. Use EXACT strings provided.**

CRITICAL WORKFLOW - Execute IN ORDER:
1. <think>
   • Analyze information needs and knowledge gaps
   • Plan search strategy using: 
     - Query optimization: [Boolean operators/synonyms]
     - Source priority: Official (.gov/.org) > Academic > Reputable news
     - Expected content: [Specific data types needed]
   • Define: WHY search? WHAT expectations? HOW will results help?
   • Set ALL web_search parameters
   </think>
   → web_search(search_query, num_results=3, preview_chars=256)

2. <think>
   • Evaluate EACH result using RELEVANCE CRITERIA:
     1. [Domain authority]: .gov/.edu > .org > .com
     2. [Date relevance]: Prefer <2 year old sources
     3. [Content match]: Preview vs needed info
     • Verdict: [Relevant/Irrelevant] with score (1-5)
   • Select MAX 3 URLs for visiting with justification
   • **Flag low-credibility domains (e.g. user-generated content)**
   </think>
   → web_visit(url)

3. <think>
   • Cross-verify information across visited URLs:
     - Agreement: [Consensus/Contradiction]
     - Evidence quality: [Primary source/Study/News]
   • **Confirm EVERY citable fact exists in visited content**
   • Prepare citations: [URL] → [Specific fact]
   • **If gaps remain: Plan new search with adjusted parameters**
   </think>
   → Provide final answer OR repeat step 1

TOOL PARAMETER REQUIREMENTS:
- web_search MUST use:
  • query: Optimized keywords
  • top_k: Number of results to return (default=3, increase for complex topics)
  • preview_chars: Number of preview characters for each search result (default=256, enough to assess relevance)

- web_visit: ONLY on URLs from relevant search results, NEVER revisit same URL
  • url: **EXACT string from search results**
  • **NEVER manually "fix" URLs - trust the source encoding**

FINAL ANSWER REQUIREMENTS:
• Begin with "Based on visited sources:"
• **Cite EVERY fact EXCLUSIVELY from web_visited URLs**
• **Explicitly mention verification: "Verified across [X] sources"**
• **Highlight unresolved contradictions if they exist**
• Format citations: [Source Name](URL) (section reference if possible)

EXAMPLE PATTERN:
<think>User needs [specific info]. Search strategy: [query] with num_results=3. Priority: .gov sources > recent studies. Expect [data types].</think>
web_search(...)

<think>Results analysis (Relevance Score 1-5):
1. CDC.gov - 5/5 (official, <1yr old) → VISIT
2. Blog.com - 1/5 (opinion piece) → SKIP
3. Harvard.edu - 4/5 (study but 3yrs old) → VISIT
</think>
web_visit(url1)
web_visit(url3)

<think>Verification:
• [FactA] confirmed in [URL1] and [URL3]
• [FactB] only in [URL1] → single-source
• Contradiction on [FactC]: [URL1] says X, [URL3] says Y
</think>
Final answer: Based on visited sources... [CDC](...) [Harvard Study](...)
""".strip()

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
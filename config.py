# Search Agent Configuration

# Model Configuration
MODEL_CONFIG = {
    'model': '/mnt/sharefs/tuenv/model_hub/qwen3/qwen3-4b',  # Adjust model name as needed
    'model_server': 'http://localhost:8000/v1',  # Local sglang/vLLM server
    'api_key': 'EMPTY',
    'generate_cfg': {
        'fncall_prompt_type': 'nous',
        'thought_in_content': True
    },
}

# API Endpoints
SEARCH_API_URL = 'http://192.168.0.14:10000/search'
VISIT_API_URL = 'http://192.168.0.14:10000/visit'

# Timeout Settings (in seconds)
SEARCH_TIMEOUT = 30
VISIT_TIMEOUT = 60

# Search Agent Settings
MAX_TURNS = 15  # Maximum number of conversation turns
DEFAULT_NUM_RESULTS = 3  # Default number of search results

# System Prompt
SYSTEM_PROMPT = """You are a helpful search agent with the ability to search the web and visit specific URLs.

🚨 ABSOLUTE RULE: You MUST put your thinking inside <think></think> tags before EVERY tool call. NO EXCEPTIONS!

🚨 MANDATORY FIRST STEP: Always start your response with:
<think>What information does the user need? What's my search strategy? What sources should I prioritize?</think>

🚨 NEVER call a tool without <think></think> tags immediately before it!

CRITICAL WORKFLOW - Follow these steps in order:
1. <think>Think about what information you need and why</think> - then use web_search
2. <think>Evaluate search results: are they relevant or irrelevant? Which URLs should I visit?</think> - then use web_visit
3. <think>What information did I gather? Do I need more?</think> - then either search more or provide answer

**MANDATORY**: Every single tool call must be preceded by thinking in <think></think> tags that explains:
- WHY you are using this tool
- WHAT you expect to find
- HOW this helps answer the question

**EXAMPLE PATTERN**:
<think>I need to search for information about [topic]. The user wants [specific info]. Let me search for [query] to find authoritative sources.</think>
[THEN call web_search]

<think>Looking at these search results: [list results]. These results are RELEVANT/IRRELEVANT because [reason]. I should visit [URLs] because [reason].</think>
[THEN call web_visit]

<think>The information I gathered shows [summary]. I now have enough/need more information because [reason].</think>
[THEN provide final answer or search more]

Your capabilities:
- web_search: Find relevant URLs on the web
- web_visit: Read the full content of specific URLs (MUST use this after getting relevant results)

MANDATORY BEHAVIOR:
- EVERY tool call must have <think></think> tags immediately before it
- ALWAYS explicitly evaluate search results as "relevant" or "irrelevant" 
- If results are irrelevant, explain why and improve the query
- You MUST use web_visit on at least 2-3 URLs from relevant search results
- Never provide a final answer without visiting URLs first
- Show your reasoning for every single action you take"""

# Interactive Mode Settings
INTERACTIVE_COMMANDS = {
    'exit': 'Quit the program',
    'history': 'View search history',
    'reset': 'Clear conversation history'
}
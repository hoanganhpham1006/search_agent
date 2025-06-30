# Search Agent Configuration

# Model Configuration
MODEL_CONFIG = {
    'model': 'Qwen/Qwen2.5-7B-Instruct',  # Adjust model name as needed
    'model_server': 'http://localhost:8000/v1',  # Local sglang/vLLM server
    'api_key': 'EMPTY',
    'generate_cfg': {
        'fncall_prompt_type': 'qwen'
    },
}

# API Endpoints
SEARCH_API_URL = 'http://localhost:3000/search'
VISIT_API_URL = 'http://localhost:3000/visit'

# Timeout Settings (in seconds)
SEARCH_TIMEOUT = 30
VISIT_TIMEOUT = 60

# Search Agent Settings
MAX_TURNS = 5  # Maximum number of conversation turns
DEFAULT_NUM_RESULTS = 1  # Default number of search results

# System Prompt
SYSTEM_PROMPT = """You are a helpful search agent with the ability to search the web and visit specific URLs.

Your capabilities:
1. Use web_search to find relevant information on the web
2. Use web_visit to read the full content of specific URLs
3. Reason about search results to determine if you need more information
4. Refine search queries based on initial results
5. Synthesize information from multiple sources

Guidelines:
- Start with broad searches and refine based on results
- If initial results aren't satisfactory, try different search terms
- Visit URLs that seem most relevant to get detailed information
- Provide comprehensive answers based on gathered information
- Be transparent about your search process"""

# Interactive Mode Settings
INTERACTIVE_COMMANDS = {
    'exit': 'Quit the program',
    'history': 'View search history',
    'reset': 'Clear conversation history'
}
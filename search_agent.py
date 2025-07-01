import json
import os
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime

from qwen_agent.llm import get_chat_model
from config import (
    MODEL_CONFIG, SEARCH_API_URL, VISIT_API_URL,
    SEARCH_TIMEOUT, VISIT_TIMEOUT, MAX_TURNS,
    DEFAULT_NUM_RESULTS, SYSTEM_PROMPT,
    INTERACTIVE_COMMANDS
)


def format_web_search_response(response_json: str) -> str:
    """Format web_search response to show indexed results with URL and Title."""
    try:
        data = json.loads(response_json)
        if 'results' in data and data['results']:
            formatted_lines = []
            for idx, result in enumerate(data['results']):
                url = result.get('url', 'N/A')
                title = result.get('metadata', {}).get('paper_title', 'No title')
                formatted_lines.append(f"Index: {idx}")
                formatted_lines.append(f"URL: {url}")
                formatted_lines.append(f"Title: {title}")
                if idx < len(data['results']) - 1:
                    formatted_lines.append("")  # Empty line between results
            return "\n".join(formatted_lines)
        else:
            return response_json
    except:
        return response_json


def format_web_visit_response(response_json: str) -> str:
    """Format web_visit response to extract only the 'data' field content."""
    try:
        data = json.loads(response_json)
        if 'data' in data:
            return data['data']
        else:
            return response_json
    except:
        return response_json


def web_search(query: str, number_results: int = DEFAULT_NUM_RESULTS) -> str:
    """
    Call the search API to search the web.
    
    Args:
        query: Search query string
        number_results: Number of results to return
        
    Returns:
        JSON string containing search results with 'url' and 'title' fields
    """
    try:
        response = requests.post(
            SEARCH_API_URL,
            json={
                'query': query,
                'number_results': number_results
            },
            timeout=SEARCH_TIMEOUT
        )
        response.raise_for_status()
        return json.dumps(response.json())
    except requests.exceptions.RequestException as e:
        error_msg = f"Error calling search API: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return json.dumps({'error': error_msg, 'results': []})


def web_visit(url: str) -> str:
    """
    Call the visit API to retrieve the full document content.
    
    Args:
        url: URL to visit
        
    Returns:
        JSON string containing the full document content
    """
    try:
        response = requests.post(
            VISIT_API_URL,
            json={'url': url},
            timeout=VISIT_TIMEOUT
        )
        response.raise_for_status()
        return json.dumps(response.json())
    except requests.exceptions.RequestException as e:
        error_msg = f"Error calling visit API: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return json.dumps({'error': error_msg, 'content': ''})


class SearchAgent:
    """
    A search agent that can perform multi-turn searches with reasoning capabilities.
    """
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the search agent.
        
        Args:
            model_config: Configuration for the LLM model
        """
        if model_config is None:
            model_config = MODEL_CONFIG
        
        self.llm = get_chat_model(model_config)
        self.conversation_history = []
        self.search_history = []
        
        # Define available functions
        self.functions = [
            {
                'name': 'web_search',
                'description': 'Search the web for information',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'The search query'
                        },
                        'number_results': {
                            'type': 'integer',
                            'description': 'Number of results to return',
                            'default': DEFAULT_NUM_RESULTS
                        }
                    },
                    'required': ['query']
                }
            },
            {
                'name': 'web_visit',
                'description': 'Visit a URL and retrieve the full document content',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'url': {
                            'type': 'string',
                            'description': 'The URL to visit'
                        }
                    },
                    'required': ['url']
                }
            }
        ]
        
        self.available_functions = {
            'web_search': web_search,
            'web_visit': web_visit
        }
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
        self.search_history = []
    
    def _add_system_prompt(self):
        """Add a system prompt to guide the agent's behavior."""
        return [{'role': 'system', 'content': SYSTEM_PROMPT}]
    
    def search(self, user_query: str, max_turns: int = MAX_TURNS) -> str:
        """
        Perform a multi-turn search based on user query.
        
        Args:
            user_query: The user's search query
            max_turns: Maximum number of conversation turns
            
        Returns:
            Final response from the agent
        """
        # Initialize conversation with system prompt and user query
        messages = self._add_system_prompt()
        messages.append({'role': 'user', 'content': user_query})
        self.conversation_history = messages.copy()
        
        print(f"\n{'='*60}")
        print(f"User Query: {user_query}")
        print(f"{'='*60}\n")
        
        final_response = ""
        turn = 0
        
        while True:  # Remove turn limitation
            turn += 1
            print(f"\n--- Turn {turn} ---")
            
            # Get response from LLM (non-streaming)
            response = self.llm.chat(
                messages=messages,
                functions=self.functions,
                stream=False
            )
            
            if not response:
                break
                
            
            # Extract reasoning and content from response
            reasoning_content = None
            regular_content = None
            
            # Handle different response formats from qwen_agent
            if isinstance(response, list) and response:
                # Check all items in the list for reasoning content and regular content
                for item in response:
                    if isinstance(item, dict):
                        if item.get('reasoning_content') and not reasoning_content:
                            reasoning_content = item.get('reasoning_content')
                        if item.get('content') and not regular_content:
                            regular_content = item.get('content')
                # Use the last message as the main response
                last_response = response[-1]
            else:
                # For function calls or other formats
                last_response = response
                if isinstance(last_response, dict):
                    reasoning_content = last_response.get('reasoning_content')
                    regular_content = last_response.get('content')
                elif hasattr(last_response, '__dict__'):
                    response_dict = getattr(last_response, '__dict__', {})
                    reasoning_content = response_dict.get('reasoning_content')
                    regular_content = response_dict.get('content')
                
            # Extract thinking from <think> tags if present in content
            thinking_text = None
            if regular_content and '<think>' in regular_content and '</think>' in regular_content:
                import re
                think_match = re.search(r'<think>(.*?)</think>', regular_content, re.DOTALL)
                if think_match:
                    thinking_text = think_match.group(1).strip()
                    # Remove the thinking from regular content
                    regular_content = regular_content.replace(think_match.group(0), '').strip()
            
            # Display thinking immediately if we have reasoning_content or extracted thinking
            if reasoning_content and reasoning_content.strip():
                print(f"\n{{'role': 'assistant', 'content': '', 'reasoning_content': '''")
                print(reasoning_content[:1500] + "..." if len(reasoning_content) > 1500 else reasoning_content)
                print("'''}}")
            elif thinking_text:
                print(f"\n{{'role': 'assistant', 'content': '', 'reasoning_content': '''")
                print(thinking_text[:1500] + "..." if len(thinking_text) > 1500 else thinking_text)
                print("'''}}")
            # Special case: if content starts with </think>, the reasoning was split
            if regular_content and regular_content.strip().startswith('</think>'):
                # Extract what comes after </think> as regular content
                after_think = regular_content.replace('</think>', '').strip()
                regular_content = after_think if after_think else ""
                # Note: The actual reasoning content is missing due to parser splitting
                
            messages.append(last_response)
            
            # Check if the model wants to call a function
            function_call_detected = False
            function_name = None
            function_args = None
            
            # Handle both standard function_call format and nous <tool_call> format
            if isinstance(last_response, dict) and last_response.get('function_call'):
                # Standard format
                function_name = last_response['function_call']['name']
                function_args = json.loads(last_response['function_call']['arguments'])
                function_call_detected = True
            elif isinstance(last_response, dict) and last_response.get('content'):
                # Check for nous format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
                content = last_response.get('content', '')
                import re
                tool_call_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL)
                if tool_call_match:
                    try:
                        tool_call_data = json.loads(tool_call_match.group(1))
                        function_name = tool_call_data.get('name')
                        function_args = tool_call_data.get('arguments', {})
                        function_call_detected = True
                    except json.JSONDecodeError:
                        print("Error parsing tool_call JSON")
            
            if function_call_detected and function_name and function_args:
                print(f"\n{{'role': 'assistant', 'content': '', 'function_call': {{'name': '{function_name}', 'arguments': '{json.dumps(function_args)}'}}}},")
                
                # Execute the function
                function_to_call = self.available_functions[function_name]
                if function_name == 'web_search':
                    function_response = function_to_call(
                        query=function_args.get('query'),
                        number_results=function_args.get('number_results', 1)
                    )
                    self.search_history.append({
                        'type': 'search',
                        'query': function_args.get('query'),
                        'timestamp': datetime.now().isoformat()
                    })
                elif function_name == 'web_visit':
                    function_response = function_to_call(
                        url=function_args.get('url')
                    )
                    self.search_history.append({
                        'type': 'visit',
                        'url': function_args.get('url'),
                        'timestamp': datetime.now().isoformat()
                    })
                
                
                # Format the response for LLM based on function type
                if function_name == 'web_search':
                    formatted_response = format_web_search_response(function_response)
                elif function_name == 'web_visit':
                    formatted_response = format_web_visit_response(function_response)
                else:
                    formatted_response = function_response

                print(f"{{'role': 'function', 'name': '{function_name}', 'content': '{formatted_response[:200]}...'}},")
                
                # Add function response to conversation
                messages.append({
                    'role': 'function',
                    'name': function_name,
                    'content': formatted_response
                })
            else:
                # Model provided a direct response (no function call)
                # Display regular content only when it's the final response
                if regular_content and regular_content.strip():
                    print(f"\n{{'role': 'assistant', 'content': '''")
                    print(regular_content)
                    print("'''}}")
                    final_response = regular_content
                elif isinstance(last_response, dict) and 'content' in last_response:
                    final_response = last_response['content']
                elif isinstance(last_response, str):
                    final_response = last_response
                
                # If we get a non-function response, it's likely a final answer
                print("\n[Model provided final response without function call]")
                break
            
            # Safety check: prevent infinite loops
            if turn > 15:
                print("\n[Stopping after 15 turns to prevent infinite loop]")
                break
                
            # Check for repeated responses (same content 2 times in a row)
            if len(messages) >= 4:
                last_content = messages[-1].get('content', '') if isinstance(messages[-1], dict) else ''
                prev_content = messages[-3].get('content', '') if isinstance(messages[-3], dict) else ''
                if last_content and last_content == prev_content and len(last_content) > 100:
                    print("\n[Stopping - detected repeated response]")
                    break
        
        self.conversation_history = messages
        return final_response
    
    def get_search_history(self) -> List[Dict[str, Any]]:
        """Get the history of searches and visits performed."""
        return self.search_history


def interactive_search():
    """Run an interactive search session."""
    agent = SearchAgent()
    
    commands_str = ', '.join([f"'{cmd}' ({desc})" for cmd, desc in INTERACTIVE_COMMANDS.items()])
    print(f"Search Agent initialized. Commands: {commands_str}")
    
    while True:
        user_input = input("\nEnter your search query: ").strip()
        
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'history':
            history = agent.get_search_history()
            print("\nSearch History:")
            for item in history:
                print(f"  - {item['type']}: {item.get('query', item.get('url'))} at {item['timestamp']}")
        elif user_input.lower() == 'reset':
            agent.reset_conversation()
            print("Conversation reset.")
        else:
            response = agent.search(user_input)
            print(f"\n{'='*60}")
            print("Final Answer:")
            print(f"{'='*60}")
            print(response)


if __name__ == '__main__':
    # Example usage
    agent = SearchAgent()
    
    # Example 1: Simple search
    # print("Example 1: Simple search")
    # response = agent.search("What are the latest developments in quantum computing?")
    # print(f"\nFinal response:\n{response}")
    
    # # Example 2: Complex multi-step search
    print("\n\nExample 2: Complex search requiring multiple steps")
    # agent.reset_conversation()
    response = agent.search("Compare the environmental impact of electric vehicles vs hydrogen fuel cell vehicles")
    print(f"\nFinal response:\n{response}")
    print("\n\nStarting interactive mode...")
    interactive_search()

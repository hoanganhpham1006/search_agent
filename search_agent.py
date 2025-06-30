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
        
        for turn in range(max_turns):
            print(f"\n--- Turn {turn + 1} ---")
            
            # Get response from LLM
            responses = []
            for response in self.llm.chat(
                messages=messages,
                functions=self.functions,
                stream=True
            ):
                responses.append(response)
            
            if not responses:
                break
                
            # Process the last complete response
            last_response = responses[-1]
            messages.append(last_response)
            
            # Check if the model wants to call a function
            if last_response.get('function_call'):
                function_name = last_response['function_call']['name']
                function_args = json.loads(last_response['function_call']['arguments'])
                
                print(f"Calling function: {function_name}")
                print(f"Arguments: {function_args}")
                
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
                
                print(f"Function response: {function_response[:200]}...")
                
                # Add function response to conversation
                messages.append({
                    'role': 'function',
                    'name': function_name,
                    'content': function_response
                })
            else:
                # Model provided a direct response
                if 'content' in last_response:
                    final_response = last_response['content']
                    print(f"\nAgent Response: {final_response[:500]}...")
                    
                    # Check if this seems like a final answer
                    if any(phrase in final_response.lower() for phrase in 
                           ['based on my search', 'in summary', 'to conclude', 
                            'the answer is', 'according to the information']):
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
    print("Example 1: Simple search")
    response = agent.search("What are the latest developments in quantum computing?")
    print(f"\nFinal response:\n{response}")
    
    # Example 2: Complex multi-step search
    print("\n\nExample 2: Complex search requiring multiple steps")
    agent.reset_conversation()
    response = agent.search("Compare the environmental impact of electric vehicles vs hydrogen fuel cell vehicles")
    print(f"\nFinal response:\n{response}")
    
    # Run interactive mode
    print("\n\nStarting interactive mode...")
    interactive_search()
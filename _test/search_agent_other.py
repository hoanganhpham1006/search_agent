import json
import os
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
import http

from qwen_agent.llm import get_chat_model
from config import (
    MODEL_CONFIG, SEARCH_API_URL, VISIT_API_URL,
    SEARCH_TIMEOUT, VISIT_TIMEOUT, MAX_TURNS,
    DEFAULT_NUM_RESULTS, SYSTEM_PROMPT,
    INTERACTIVE_COMMANDS
)


def format_web_search_response(data) -> str:
    """Format web_search response to show indexed results with URL and Title."""
    try:
        if isinstance(data, dict) and 'results' in data and data['results']:
            formatted_lines = []
            for idx, result in enumerate(data['results']):
                url = result.get('url', 'N/A')
                title = result.get('metadata', {}).get('paper_title', 'No title')
                if title.endswith("..."):
                    title = title[:-3]  # Remove trailing "..."
                preview = result.get('preview', 'No preview available')
                if preview.endswith("..."):
                    preview = preview[:-3]  # Remove trailing "..."
                formatted_lines.append(f"Index: {idx}")
                formatted_lines.append(f"URL: {url}")
                formatted_lines.append(f"Title: {title}")
                formatted_lines.append(f"Preview: {preview}")
                if idx < len(data['results']) - 1:
                    formatted_lines.append("")  # Empty line between results
            return "\n".join(formatted_lines)
        else:
            return str(data)
    except Exception as e:
        return f"Error formatting web_search response: {str(e)}"


def format_web_visit_response(data) -> str:
    """Format web_visit response to extract only the 'data' field content."""
    try:
        if isinstance(data, dict) and 'data' in data:
            return data['data']
        elif isinstance(data, str):
            return data
        else:
            return str(data)
    except Exception as e:
        return f"Error formatting web_visit response: {str(e)}"


def web_search(query: str, min_results: int = 3, max_retries: int = 3) -> dict:
    """Perform a web search using Serper API with retry logic to ensure minimum results."""
    import time
    
    for attempt in range(max_retries):
        conn = None
        try:
            conn = http.client.HTTPSConnection("google.serper.dev", timeout=30)
            payload = json.dumps({
                "q": query,
                "num": 10
            })
            headers = {
            }

            conn.request("POST", "/search", payload, headers)
            res = conn.getresponse()
            raw_data = res.read()
            
            # Check if response is valid
            if res.status != 200:
                if attempt == max_retries - 1:
                    return {'results': [], 'error': f"HTTP Error {res.status}: {res.reason}"}
                time.sleep(1)  # Wait before retry
                continue
            
            result = json.loads(raw_data.decode("utf-8"))

            # Format the results
            formatted_results = {
                'results': []
            }
            if "organic" in result and result["organic"]:
                for item in result["organic"]:
                    snippet = item.get('snippet', 'No preview available')
                        
                    formatted_results['results'].append({
                        'url': item.get("link", ""),
                        'metadata': {
                            'paper_title': item.get('title', 'No title')
                        },
                        'preview': snippet
                    })    
            
            # Check if we have enough results
            if len(formatted_results['results']) >= min_results:
                return formatted_results
            elif attempt < max_retries - 1:
                print(f"Only got {len(formatted_results['results'])} results, retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(1)  # Wait before retry
                continue
            else:
                # Last attempt, return whatever we got
                return formatted_results
                
        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                return {'results': [], 'error': f"JSON parsing error: {str(e)}"}
            time.sleep(1)
            continue
        except Exception as e:
            if attempt == max_retries - 1:
                return {'results': [], 'error': f"Error performing web search: {str(e)}"}
            time.sleep(1)
            continue
        finally:
            if conn:
                conn.close()
    
    return {'results': [], 'error': f"Failed to get {min_results} results after {max_retries} attempts"}
    

def clean_web_content(content: str) -> str:
    """Clean web content by removing URLs, extra whitespace, and limiting length."""
    import re
    
    if not content:
        return ""
    
    # Remove URLs (http/https/ftp/www links)
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]*|www\.[^\s<>"{}|\\^`\[\]]*|ftp://[^\s<>"{}|\\^`\[\]]*'
    content = re.sub(url_pattern, '', content)
    
    # Remove email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    content = re.sub(email_pattern, '', content)
    
    # Remove markdown links [text](url)
    markdown_link_pattern = r'\[([^\]]*)\]\([^)]*\)'
    content = re.sub(markdown_link_pattern, r'\1', content)
    
    # Remove excessive whitespace, newlines, and special characters
    content = re.sub(r'\s+', ' ', content)  # Replace multiple whitespace with single space
    content = re.sub(r'\n+', '\n', content)  # Replace multiple newlines with single newline
    content = re.sub(r'[\t\r\f\v]+', ' ', content)  # Replace tabs and other whitespace
    
    # Remove common navigation/footer text patterns
    noise_patterns = [
        r'Skip to main content',
        r'Privacy Policy',
        r'Terms of Service',
        r'Cookie Policy',
        r'© \d{4}',
        r'All rights reserved',
        r'Subscribe to newsletter',
        r'Follow us on',
        r'Share this',
        r'Print this page'
    ]
    
    for pattern in noise_patterns:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE)
    
    # Strip leading/trailing whitespace
    content = content.strip()
    
    # Cap at 10000 characters
    if len(content) > 10000:
        content = content[:10000] + "\n[Content truncated to 10,000 characters]"
    
    return content

def web_visit(url: str) -> dict:
    """Visit a webpage and extract its content using Serper API."""
    conn = None
    try:
        conn = http.client.HTTPSConnection("scrape.serper.dev", timeout=30)
        payload = json.dumps({
            "url": url,
            "includeMarkdown": True
        })
        header = {
        }

        conn.request("POST", "/", payload, headers)
        res = conn.getresponse()
        raw_data = res.read()
        
        # Check if response is valid
        if res.status != 200:
            return {'data': f"HTTP Error {res.status}: {res.reason}"}
        
        result = json.loads(raw_data.decode("utf-8"))

        # Extract text content
        raw_content = ""
        if "markdown" in result and result["markdown"]:
            raw_content = result["markdown"]
        elif "text" in result and result["text"]:
            raw_content = result["text"]
        else:
            raw_content = json.dumps(result, indent=2)

        # Clean and process the content
        cleaned_content = clean_web_content(raw_content)
        
        return {'data': cleaned_content}
    except json.JSONDecodeError as e:
        return {'data': f"JSON parsing error: {str(e)}"}
    except Exception as e:
        return {'data': f"Error visiting webpage: {str(e)}"}
    finally:
        if conn:
            conn.close()


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
        self.web_search_count = 0
        self.web_visit_count = 0
        
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
        self.web_search_count = 0
        self.web_visit_count = 0
    
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
        user_query += '\n\nPlease reason step-by-step, and put your final answer within \\boxed{}.'
        messages.append({'role': 'user', 'content': user_query})
        self.conversation_history = messages.copy()
        
        print(f"\n{'='*60}")
        print(f"User Query: {user_query}")
        print(f"{'='*60}\n")
        
        final_response = ""
        turn = 0
        
        while True:
            # Get response from LLM (non-streaming)
            response = self.llm.chat(
                messages=messages,
                functions=self.functions,
                # functions=[{"type": "function", "function": x} for x in self.functions],
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
            # Now we have valid content, increment turn counter
            turn += 1
            print(f"\n--- Turn {turn} ---")
            
            # Display thinking immediately if we have reasoning_content
            if reasoning_content and reasoning_content.strip():
                print(f"\n{{'role': 'assistant', 'content': '', 'reasoning_content': '''")
                print(reasoning_content[:1500] + "..." if len(reasoning_content) > 1500 else reasoning_content)
                print("'''}}")
            # import pdb;pdb.set_trace()
            messages.append(last_response)
            
            # Check if the model wants to call functions
            tool_calls = []
            
            # Handle multiple function call formats
            if isinstance(last_response, dict):
                # Check for standard tool_calls format (multiple calls with IDs)
                if last_response.get('tool_calls'):
                    tool_calls = last_response['tool_calls']
                # Check for single function_call format
                elif last_response.get('function_call'):
                    # Convert single function call to tool_calls format
                    function_name = last_response['function_call']['name']
                    function_args = json.loads(last_response['function_call']['arguments'])
                    tool_calls = [{
                        'id': f'call_{turn}_{0}',
                        'function': {
                            'name': function_name,
                            'arguments': json.dumps(function_args)
                        },
                        'type': 'function'
                    }]
                # Check for content-based formats
                elif last_response.get('content'):
                    content = last_response.get('content', '')
                    import re
                    import uuid
                    
                    # Check for Kimi format: <|tool_calls_section_begin|><|tool_call_begin|>functions.func_name:0<|tool_call_argument_begin|>{...}<|tool_call_end|><|tool_calls_section_end|>
                    kimi_matches = re.findall(r'<\|tool_call_begin\|>functions\.(\w+):(\d+)<\|tool_call_argument_begin\|>(\{.*?\})<\|tool_call_end\|>', content, re.DOTALL)
                    if kimi_matches:
                        for idx, (func_name, call_idx, args_str) in enumerate(kimi_matches):
                            try:
                                function_args = json.loads(args_str)
                                tool_calls.append({
                                    'id': f'call_{turn}_{idx}',
                                    'function': {
                                        'name': func_name,
                                        'arguments': json.dumps(function_args)
                                    },
                                    'type': 'function'
                                })
                            except json.JSONDecodeError:
                                print(f"Error parsing Kimi tool_call JSON for function {func_name}")
                        
                        # Remove the entire Kimi tool call section from content
                        if tool_calls:
                            kimi_section_pattern = r'<\|tool_calls_section_begin\|>.*?<\|tool_calls_section_end\|>'
                            cleaned_content = re.sub(kimi_section_pattern, '', content, flags=re.DOTALL).strip()
                            if isinstance(last_response, dict):
                                last_response['content'] = cleaned_content
                    else:
                        # Check for nous format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
                        tool_call_matches = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL)
                        if tool_call_matches:
                            for idx, tool_call_match in enumerate(tool_call_matches):
                                try:
                                    tool_call_data = json.loads(tool_call_match)
                                    function_name = tool_call_data.get('name')
                                    function_args = tool_call_data.get('arguments', {})
                                    tool_calls.append({
                                        'id': f'call_{turn}_{idx}',
                                        'function': {
                                            'name': function_name,
                                            'arguments': json.dumps(function_args, sort_keys=True)
                                        },
                                        'type': 'function'
                                    })
                                except json.JSONDecodeError:
                                    print("Error parsing nous tool_call JSON")
            
            # Process all tool calls
            if tool_calls:
                # Deduplicate tool calls - keep only first occurrence of same function with same args
                seen_calls = set()
                deduplicated_tool_calls = []
                
                for tool_call in tool_calls:
                    function_info = tool_call.get('function', {})
                    function_name = function_info.get('name')
                    function_args_str = function_info.get('arguments', '{}')
                    
                    # Create a signature for this tool call
                    call_signature = (function_name, function_args_str)
                    
                    if call_signature not in seen_calls:
                        seen_calls.add(call_signature)
                        deduplicated_tool_calls.append(tool_call)
                    else:
                        print(f"\n[Skipping duplicate tool call: {function_name} with args {function_args_str}]")
                
                # Execute ALL unique functions and add their responses
                for tool_call in deduplicated_tool_calls:
                    function_info = tool_call.get('function', {})
                    function_name = function_info.get('name')
                    function_args_str = function_info.get('arguments', '{}')
                    
                    # Print tool call for WebUI detection
                    print(f"\n{{'role': 'assistant', 'content': '', 'function_call': {{'name': '{function_name}', 'arguments': '{function_args_str}'}}}},")
                    
                    try:
                        function_args = json.loads(function_args_str)
                    except json.JSONDecodeError:
                        function_args = {}
                    
                    # Execute the function
                    function_to_call = self.available_functions.get(function_name)
                    if function_to_call:
                        if function_name == 'web_search':
                            function_response = function_to_call(
                                query=function_args.get('query'),
                            )
                            self.web_search_count += 1
                            self.search_history.append({
                                'type': 'search',
                                'query': function_args.get('query'),
                                'timestamp': datetime.now().isoformat()
                            })
                        elif function_name == 'web_visit':
                            function_response = function_to_call(
                                url=function_args.get('url')
                            )
                            self.web_visit_count += 1
                            self.search_history.append({
                                'type': 'visit',
                                'url': function_args.get('url'),
                                'timestamp': datetime.now().isoformat()
                            })
                        
                        # Format the response based on function type
                        if function_name == 'web_search':
                            formatted_response = format_web_search_response(function_response)
                        elif function_name == 'web_visit':
                            formatted_response = format_web_visit_response(function_response)
                        else:
                            formatted_response = str(function_response)
                        
                        # Add function response message
                        function_response_msg = {
                            'role': 'function',
                            'name': function_name,
                            'content': formatted_response
                        }
                        messages.append(function_response_msg)
                        print(f"{{'role': 'function', 'name': '{function_name}', 'content': '{formatted_response[:500]}...'}},")
                    else:
                        print(f"Warning: Function {function_name} not found in available_functions")
            else:
                # Model provided a direct response (no function call) - this is the final answer
                if regular_content and regular_content.strip():
                    print(f"\n{{'role': 'assistant', 'content': '''")
                    print(regular_content)
                    print("'''}}")
                    final_response = regular_content
                elif isinstance(last_response, dict) and 'content' in last_response:
                    final_response = last_response['content']
                elif isinstance(last_response, str):
                    final_response = last_response
                
                # Check for unreliable answer: zero web_search OR zero web_visit before final answer
                # min_web_search = 1  # Configurable threshold
                # min_web_visit = 0   # Configurable threshold
                
                # if self.web_search_count < min_web_search or self.web_visit_count < min_web_visit:
                #     print(f"\n[UNRELIABLE ANSWER DETECTED: web_search_count={self.web_search_count} (min={min_web_search}), web_visit_count={self.web_visit_count} (min={min_web_visit}) - Re-running turn]")
                    
                #     # Add a retry counter to prevent infinite loops
                #     reliability_retry_key = "reliability_retries"  # Fixed key, not dependent on turn
                #     if not hasattr(self, '_reliability_retry_counts'):
                #         self._reliability_retry_counts = {}
                #     self._reliability_retry_counts[reliability_retry_key] = self._reliability_retry_counts.get(reliability_retry_key, 0) + 1
                    
                #     if self._reliability_retry_counts[reliability_retry_key] > 10:
                #         print(f"\n[Max reliability retries reached ({self._reliability_retry_counts[reliability_retry_key]}), accepting potentially unreliable answer]")
                #         # Reset retry counter and continue with final answer
                #         self._reliability_retry_counts[reliability_retry_key] = 0
                #     else:
                #         # Remove the last assistant message and continue to retry the turn
                #         messages.pop()  # Remove the unreliable response
                #         # Add a guidance message to encourage proper tool usage
                #         # guidance_msg = {
                #         #     'role': 'user', 
                #         #     'content': f'Please ensure you perform adequate research before providing a final answer. You must use web_search (called {self.web_search_count} times, minimum {min_web_search}) and web_visit (called {self.web_visit_count} times, minimum {min_web_visit}) to gather reliable information.'
                #         # }
                #         # messages.append(guidance_msg)
                #         continue  # Retry this turn
                
                # No function call means final answer (and it's reliable)
                print("\n[Model provided final response]")
                break
            
            # Safety check: prevent infinite loops
            if turn > 50:
                print("\n[Stopping after 50 turns to prevent infinite loop]")
                break
                
            # Check for repeated responses (same content 2 times in a row) - only for assistant messages
            assistant_messages = [msg for msg in messages if isinstance(msg, dict) and msg.get('role') == 'assistant']
            if len(assistant_messages) >= 2:
                last_assistant_content = assistant_messages[-1].get('content', '')
                prev_assistant_content = assistant_messages[-2].get('content', '')
                if (last_assistant_content and prev_assistant_content and 
                    last_assistant_content == prev_assistant_content and 
                    len(last_assistant_content) > 100):
                    print("\n[Stopping - detected repeated assistant response]")
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
    response = agent.search("Compare the environmental impact of electric vehicles vs hydrogen fuel cell vehicles.")
    print(f"\nFinal response:\n{response}")
    print("\n\nStarting interactive mode...")
    interactive_search()

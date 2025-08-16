import json
import os
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
import http
import re
import uuid

from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role,
    Message,
    Conversation,
    DeveloperContent,
    SystemContent,
    TextContent,
    ToolDescription,
    Author
)
from openai import OpenAI

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
        headers = {
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
    Uses OpenAI Harmony for GPT-OSS models.
    """
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the search agent.
        
        Args:
            model_config: Configuration for the GPT-OSS model
        """
        if model_config is None:
            model_config = MODEL_CONFIG
        
        # Initialize OpenAI client for GPT-OSS
        self.client = OpenAI(
            base_url=model_config.get('model_server', 'http://localhost:1234/v1'),
            api_key=model_config.get('api_key', 'EMPTY')
        )
        
        # Initialize Harmony encoding for GPT-OSS
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        
        self.model_config = model_config
        self.conversation_history = []
        self.search_history = []
        self.web_search_count = 0
        self.web_visit_count = 0
        
        # Define available functions in Harmony format
        self.functions = [
            ToolDescription.new(
                "web_search",
                "Search the web for information",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            ),
            ToolDescription.new(
                "web_visit",
                "Visit a URL and retrieve the full document content", 
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to visit"
                        }
                    },
                    "required": ["url"]
                }
            )
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
    
    def _create_harmony_conversation(self, messages: List[Dict]) -> Conversation:
        """Create a Harmony conversation from standard message format."""
        harmony_messages = []
        
        # Add system message
        system_msg = Message.from_role_and_content(
            Role.SYSTEM, 
            SystemContent()
        )
        harmony_messages.append(system_msg)
        
        # Add developer message with tool definitions
        developer_msg = Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent()
            .with_instructions(SYSTEM_PROMPT)
            .with_function_tools(self.functions)
        )
        harmony_messages.append(developer_msg)
        
        # Convert other messages
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content', '')
            
            if role == 'user':
                harmony_messages.append(
                    Message.from_role_and_content(Role.USER, TextContent.model_construct(text=content))
                )
            elif role == 'assistant':
                harmony_messages.append(
                    Message.from_role_and_content(Role.ASSISTANT, TextContent.model_construct(text=content))
                )
            elif role == 'function':
                # Convert function response to tool response
                harmony_messages.append(
                    Message.from_author_and_content(
                        Author.model_construct(role=Role.TOOL, name=f"functions.{msg.get('name')}"),
                        TextContent.model_construct(text=content)
                    ).with_recipient("assistant")
                )
        
        return Conversation.from_messages(harmony_messages)
    
    def _parse_harmony_response(self, response_text: str) -> Dict:
        """Parse tool calls from Harmony-formatted response."""
        tool_calls = []
        
        # Look for function calls in Harmony format
        # Check for commentary channel with JSON
        json_match = re.search(r'\{[^}]*"(web_search|web_visit)"[^}]*\}', response_text, re.DOTALL)
        if json_match:
            try:
                tool_data = json.loads(json_match.group())
                if 'web_search' in response_text:
                    return {
                        'tool_calls': [{
                            'id': f'harmony_call_{uuid.uuid4().hex[:8]}',
                            'function': {
                                'name': 'web_search',
                                'arguments': json.dumps({'query': tool_data.get('query', tool_data.get('web_search', ''))})
                            },
                            'type': 'function'
                        }],
                        'content': response_text
                    }
                elif 'web_visit' in response_text:
                    return {
                        'tool_calls': [{
                            'id': f'harmony_call_{uuid.uuid4().hex[:8]}',
                            'function': {
                                'name': 'web_visit', 
                                'arguments': json.dumps({'url': tool_data.get('url', tool_data.get('web_visit', ''))})
                            },
                            'type': 'function'
                        }],
                        'content': response_text
                    }
            except:
                pass
        
        # Check for standard tool call formats
        tool_call_matches = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response_text, re.DOTALL)
        if tool_call_matches:
            for idx, tool_call_match in enumerate(tool_call_matches):
                try:
                    tool_call_data = json.loads(tool_call_match)
                    function_name = tool_call_data.get('name')
                    function_args = tool_call_data.get('arguments', {})
                    tool_calls.append({
                        'id': f'harmony_call_{uuid.uuid4().hex[:8]}',
                        'function': {
                            'name': function_name,
                            'arguments': json.dumps(function_args)
                        },
                        'type': 'function'
                    })
                except:
                    pass
        
        return {
            'tool_calls': tool_calls,
            'content': response_text
        }
    
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
            # Create Harmony conversation
            convo = self._create_harmony_conversation(messages[1:])  # Skip system message
            
            try:
                # Use the responses API for GPT-OSS with tool use capabilities
                # Convert to simple message format that responses API expects
                input_messages = []
                for msg in messages:
                    if msg['role'] in ['user', 'system']:
                        input_messages.append({
                            'role': msg['role'], 
                            'content': msg['content']
                        })
                    elif msg['role'] == 'assistant':
                        # For assistant messages, preserve tool calls if present
                        assistant_msg = {
                            'role': msg['role'], 
                            'content': msg['content']
                        }
                        # Include tool_calls in the message so model can see what it called
                        if 'tool_calls' in msg:
                            assistant_msg['tool_calls'] = msg['tool_calls']
                        input_messages.append(assistant_msg)
                    elif msg['role'] == 'function':
                        # Convert function responses to user messages 
                        # The responses API doesn't seem to accept 'tool' role
                        input_messages.append({
                            'role': 'user',
                            'content': msg['content']
                        })
                # Add your custom tools in the correct format for responses API
                response = self.client.responses.create(
                    input=input_messages,
                    model=self.model_config.get('model'),
                    tools=[{
                        'type': 'function',
                        'name': 'web_search',
                        'description': 'Search the web for information',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'query': {'type': 'string', 'description': 'The search query'}
                            },
                            'required': ['query']
                        }
                    }, {
                        'type': 'function',
                        'name': 'web_visit',
                        'description': 'Visit a URL and retrieve the full document content',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'url': {'type': 'string', 'description': 'The URL to visit'}
                            },
                            'required': ['url']
                        }
                    }],
                    temperature=self.model_config.get('generate_cfg', {}).get('temperature', 0.7),
                    max_output_tokens=self.model_config.get('generate_cfg', {}).get('max_tokens', 4096)
                )
                
                # Parse the responses API output - it returns a Response object with output array
                reasoning_content = None
                regular_content = ''
                tool_calls = []
                
                if hasattr(response, 'output') and response.output:
                    for item in response.output:
                        if hasattr(item, 'type'):
                            if item.type == 'reasoning':
                                # Extract reasoning content
                                if hasattr(item, 'content') and item.content:
                                    reasoning_texts = []
                                    for content_item in item.content:
                                        if content_item.get('type') == 'reasoning_text':
                                            reasoning_texts.append(content_item.get('text'))
                                    reasoning_content = '\n'.join(reasoning_texts)
                            
                            elif item.type == 'function_call':
                                # Extract tool call
                                tool_calls.append({
                                    'id': item.call_id,
                                    'function': {
                                        'name': item.name,
                                        'arguments': item.arguments
                                    },
                                    'type': 'function'
                                })
                            
                            elif item.type == 'message':
                                # Extract regular text content
                                if hasattr(item, 'content') and item.content:
                                    regular_texts = []
                                    for content_item in item.content:
                                        if content_item.type == 'output_text':
                                            regular_texts.append(content_item.text)
                                    regular_content = '\n'.join(regular_texts)

                # If no content found, break
                if not reasoning_content and not regular_content and not tool_calls:
                    break
                
                # Now we have valid content, increment turn counter
                turn += 1
                print(f"\n--- Turn {turn} ---")
                
                # Display thinking immediately if we have reasoning_content
                if reasoning_content and reasoning_content.strip():
                    print(f"\n{{'role': 'assistant', 'content': '', 'reasoning_content': '''")
                    print(reasoning_content[:1500] + "..." if len(reasoning_content) > 1500 else reasoning_content)
                    print("'''}}")
                
                # Create assistant message
                assistant_msg = {'role': 'assistant', 'content': regular_content}
                if tool_calls:
                    assistant_msg['tool_calls'] = tool_calls
                
                messages.append(assistant_msg)
                
                # Process tool calls (same logic as original)
                if tool_calls:
                    # Deduplicate tool calls
                    seen_calls = set()
                    deduplicated_tool_calls = []
                    
                    for tool_call in tool_calls:
                        function_info = tool_call.get('function', {})
                        function_name = function_info.get('name')
                        function_args_str = function_info.get('arguments', '{}')
                        
                        call_signature = (function_name, function_args_str)
                        
                        if call_signature not in seen_calls:
                            seen_calls.add(call_signature)
                            deduplicated_tool_calls.append(tool_call)
                        else:
                            print(f"\n[Skipping duplicate tool call: {function_name} with args {function_args_str}]")
                    
                    # Execute functions
                    for tool_call in deduplicated_tool_calls:
                        function_info = tool_call.get('function', {})
                        function_name = function_info.get('name')
                        function_args_str = function_info.get('arguments', '{}')
                        
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
                            
                            # Format response
                            if function_name == 'web_search':
                                formatted_response = format_web_search_response(function_response)
                            elif function_name == 'web_visit':
                                formatted_response = format_web_visit_response(function_response)
                            else:
                                formatted_response = str(function_response)
                            
                            # Add function response message with call_id for proper matching
                            function_response_msg = {
                                'role': 'function',
                                'name': function_name,
                                'content': formatted_response,
                                'tool_call_id': tool_call.get('id', f'call_{function_name}')
                            }
                            messages.append(function_response_msg)
                            print(f"{{'role': 'function', 'name': '{function_name}', 'content': '{formatted_response[:500]}...'}},")
                        else:
                            print(f"Warning: Function {function_name} not found in available_functions")
                else:
                    # Model provided direct response - final answer
                    if regular_content and regular_content.strip():
                        print(f"\n{{'role': 'assistant', 'content': '''")
                        print(regular_content)
                        print("'''}}")
                        final_response = regular_content
                    
                    print("\n[Model provided final response]")
                    break
                
            except Exception as e:
                print(f"Error during API call: {str(e)}")
                break
            
            # Safety checks (same as original)
            if turn > 50:
                print("\n[Stopping after 50 turns to prevent infinite loop]")
                break
                
            # Check for repeated responses
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
    
    # Example: Complex multi-step search
    print("\n\nExample: Complex search requiring multiple steps")
    response = agent.search("Compare the environmental impact of electric vehicles vs hydrogen fuel cell vehicles.")
    print(f"\nFinal response:\n{response}")
    print("\n\nStarting interactive mode...")
    interactive_search()

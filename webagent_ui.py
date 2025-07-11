#!/usr/bin/env python3

import argparse
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.gui import WebUI
from qwen_agent.tools import BaseTool, TOOL_REGISTRY
from qwen_agent.llm.schema import Message

# Import our working search agent
from search_agent import SearchAgent as OriginalSearchAgent

# Import configuration from our existing config.py
from config import (
    MODEL_CONFIG, SEARCH_API_URL, VISIT_API_URL,
    SEARCH_TIMEOUT, VISIT_TIMEOUT, MAX_TURNS,
    DEFAULT_NUM_RESULTS, SYSTEM_PROMPT,
    INTERACTIVE_COMMANDS
)


import io
import sys
import threading
import queue
import time


class OutputCapture:
    """Captures stdout output in a thread-safe way."""
    def __init__(self):
        self.queue = queue.Queue()
        self.original_stdout = sys.stdout
        self.capture_buffer = io.StringIO()
        
    def write(self, text):
        self.original_stdout.write(text)  # Still print to terminal
        self.capture_buffer.write(text)
        if '\n' in text:  # Complete line
            line = self.capture_buffer.getvalue()
            self.capture_buffer = io.StringIO()
            self.queue.put(line)
    
    def flush(self):
        self.original_stdout.flush()
    
    def get_output(self, timeout=0.1):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None


class SearchAgentForWebUI:
    """Wrapper that captures SearchAgent output and streams to WebUI."""
    
    def __init__(self, name, description):
        self.search_agent = OriginalSearchAgent()
        self.name = name
        self.description = description
        self.function_map = {}
    
    def run(self, messages, **kwargs):
        """Run method that captures and streams real execution."""
        if not messages:
            return
        
        # Get the user message
        user_message = None
        for msg in reversed(messages):
            if isinstance(msg, dict):
                role = msg.get('role')
                content = msg.get('content')
                if role == 'user':
                    if isinstance(content, list):
                        user_message = content[0].get('text', '') if content else ''
                    else:
                        user_message = content or ''
                    break
            elif hasattr(msg, 'role') and msg.role == 'user':
                if hasattr(msg, 'content'):
                    if isinstance(msg.content, list):
                        user_message = msg.content[0].get('text', '') if msg.content else ''
                    else:
                        user_message = msg.content or ''
                    break
        
        if not user_message:
            return
        
        response = []
        
        # Create output capture
        capture = OutputCapture()
        
        # Run search agent in a thread to capture output
        result_container = {'final_result': None, 'error': None}
        
        def run_search():
            old_stdout = sys.stdout
            sys.stdout = capture
            try:
                result_container['final_result'] = self.search_agent.search(user_message)
            except Exception as e:
                result_container['error'] = str(e)
            finally:
                sys.stdout = old_stdout
        
        # Start search in thread
        search_thread = threading.Thread(target=run_search)
        search_thread.start()
        
        # Buffer to collect multi-line output
        output_buffer = []
        reasoning_mode = False
        content_mode = False
        
        # Stream output as it comes
        while search_thread.is_alive() or not capture.queue.empty():
            output = capture.get_output(timeout=0.1)
            if output:
                output = output.strip()
                
                # Check for reasoning start
                if "{'role': 'assistant', 'content': '', 'reasoning_content': '''" in output:
                    reasoning_mode = True
                    output_buffer = []
                    continue
                
                # Check for reasoning end
                if reasoning_mode and "'''}}" in output:
                    reasoning_mode = False
                    # Join all collected lines
                    reasoning_text = '\n'.join(output_buffer)
                    if reasoning_text:
                        msg = {
                            'role': 'assistant',
                            'content': f"**🧠 Thinking:**\n\n{reasoning_text}",
                            'name': self.name
                        }
                        response.append(msg)
                        yield response.copy()
                    output_buffer = []
                    continue
                
                # Collect reasoning lines
                if reasoning_mode and output:
                    output_buffer.append(output)
                    continue
                    
                # Handle tool calls
                if "{'role': 'assistant', 'content': '', 'function_call':" in output:
                    # This is a tool call
                    tool_info = self._extract_tool_from_output(output)
                    if tool_info:
                        msg = {
                            'role': 'assistant',
                            'content': f" \n\n**🔧 Start tool calling:** `{tool_info['name']}`\n\nArguments:\n```json\n{tool_info['args']}\n```",
                            'name': self.name
                        }
                        response.append(msg)
                        yield response.copy()
                        
                elif "{'role': 'function', 'name':" in output:
                    # This is a tool result
                    tool_result = self._extract_tool_result_from_output(output)
                    if tool_result:
                        msg = {
                            'role': 'assistant',
                            'content': f"**✅ Finish tool calling:** `{tool_result['name']}`\n\nResult preview:\n{tool_result['content'][:300]}...",
                            'name': self.name
                        }
                        response.append(msg)
                        yield response.copy()
                        
                elif "{'role': 'assistant', 'content': '''" in output:
                    # Start collecting content
                    content_mode = True
                    output_buffer = []
                    continue
                
                # Collect content lines
                if content_mode and output:
                    output_buffer.append(output)
                    continue
        
        # Wait for thread to complete
        search_thread.join(timeout=60)
        
        # Show final result
        if result_container['final_result']:
            final_msg = {
                'role': 'assistant',
                'content': f" \n\n**📊 Final Answer:**\n\n{result_container['final_result']}",
                'name': self.name
            }
            response.append(final_msg)
            yield response.copy()
        elif result_container['error']:
            error_msg = {
                'role': 'assistant',
                'content': f"\n\n**❌ Error:**\n\n{result_container['error']}",
                'name': self.name
            }
            response.append(error_msg)
            yield response.copy()
    
    def _extract_reasoning_from_output(self, output):
        """Extract reasoning content from printed output."""
        try:
            # Look for the reasoning content between triple quotes
            import re
            match = re.search(r"reasoning_content': '''(.+?)'''}", output, re.DOTALL)
            if match:
                return match.group(1).strip()
        except:
            pass
        return None
    
    def _extract_tool_from_output(self, output):
        """Extract tool call info from printed output."""
        try:
            import re
            import json
            match = re.search(r"'function_call': \{'name': '(.+?)', 'arguments': '(.+?)'\}\}", output)
            if match:
                name = match.group(1)
                args_str = match.group(2)
                # Parse the JSON arguments
                args = json.loads(args_str)
                return {'name': name, 'args': json.dumps(args, indent=2)}
        except:
            pass
        return None
    
    def _extract_tool_result_from_output(self, output):
        """Extract tool result from printed output."""
        try:
            import re
            match = re.search(r"'role': 'function', 'name': '(.+?)', 'content': '(.+?)'", output)
            if match:
                name = match.group(1)
                content = match.group(2)
                return {'name': name, 'content': content}
        except:
            pass
        return None
    
    def _extract_final_content_from_output(self, output):
        """Extract final content from printed output."""
        try:
            # Look for content between triple quotes
            import re
            match = re.search(r"'content': '''(.+?)'''}", output, re.DOTALL)
            if match:
                return match.group(1).strip()
        except:
            pass
        return None


def init_search_agent_service(
    name: str = 'Search Agent Assistant', 
    reasoning: bool = True, 
    max_llm_calls: int = 50, 
    tools = ['web_search', 'web_visit']
):
    """Initialize the search agent service using our working SearchAgent."""
    
    # Use our working SearchAgent wrapped for WebUI compatibility
    agent = SearchAgentForWebUI(
        name=name,
        description='Your small and intelligent search agent that can search the web and visit URLs to provide comprehensive answers'
    )
    
    return agent


def app_gui():
    """Run the Web UI application using exact WebAgent demo pattern."""
    
    print("=" * 60)
    print("🔍 Search Agent Assistant (WebAgent Pattern)")
    print("=" * 60)
    print(f"Model: {MODEL_CONFIG['model']}")
    print(f"Model Server: {MODEL_CONFIG['model_server']}")
    print(f"Search API: {SEARCH_API_URL}")
    print(f"Visit API: {VISIT_API_URL}")
    print("=" * 60)
    
    # Initialize agents exactly like WebAgent demo
    agents = []
    for name, reasoning, max_llm_calls, tools in [
        ('II-Searcher-4B', True, 50, ['web_search', 'web_visit'])
    ]:
        search_agent = init_search_agent_service(
            name=name,
            reasoning=reasoning,
            max_llm_calls=max_llm_calls,
            tools=tools
        )
        agents.append(search_agent)
    
    # Configure chatbot exactly like WebAgent demo
    chatbot_config = {
        'prompt.suggestions': [
            "What are the latest developments in quantum computing?",
            "Compare the environmental impact of electric vehicles vs hydrogen fuel cell vehicles", 
            "Explain the current state of artificial intelligence research",
            "What are the health benefits and risks of intermittent fasting?",
            "How does blockchain technology work and what are its applications?",
            "What are the causes and solutions for climate change?",
            "Explain the latest discoveries in space exploration",
            "What are the pros and cons of renewable energy sources?"
        ],
        'user.name': 'User',
        'verbose': True
    }
    
    # Use exactly the same pattern as WebAgent demo
    WebUI(
        agent=agents,  # Pass list of agents like WebAgent
        chatbot_config=chatbot_config,
    ).run(
        share=True,
        server_name='0.0.0.0',
        server_port=7860,
        concurrency_limit=20,
        enable_mention=False,
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Search Agent Assistant with Final WebAgent UI')
    parser.add_argument('--port', type=int, default=7860, help='Port for Web UI')
    parser.add_argument('--share', action='store_true', default=True, help='Enable Gradio sharing')
    
    args = parser.parse_args()
    
    if args.port != 7860:
        agent = init_search_agent_service()
        chatbot_config = {
            'prompt.suggestions': [
                "What are the latest developments in quantum computing?",
                "Compare the environmental impact of electric vehicles vs hydrogen fuel cell vehicles", 
                "Explain the current state of artificial intelligence research",
                "What are the health benefits and risks of intermittent fasting?",
                "How does blockchain technology work and what are its applications?",
                "What are the causes and solutions for climate change?",
                "Explain the latest discoveries in space exploration",
                "What are the pros and cons of renewable energy sources?"
            ],
            'user.name': 'User',
            'verbose': True
        }
        
        WebUI(agent=agent, chatbot_config=chatbot_config).run(
            share=args.share, server_name='0.0.0.0', server_port=args.port,
            concurrency_limit=10, enable_mention=False
        )
    else:
        app_gui()


if __name__ == '__main__':
    main()
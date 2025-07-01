#!/usr/bin/env python3
"""
Benchmark script for the Search Agent with full trace capture.

This script evaluates the search agent's performance on a set of questions,
capturing detailed traces including model thinking, function calls, and responses.
"""

import json
import time
import pandas as pd
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import argparse
from io import StringIO
import multiprocessing as mp
from multiprocessing import Pool, Queue, Manager
import threading
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import requests

from search_agent import SearchAgent
from config import MODEL_CONFIG
from benchmark_config import (
    BENCHMARK_MODEL_CONFIG, BENCHMARK_BASE_HOST, AVAILABLE_PORTS,
    BENCHMARK_SEARCH_TIMEOUT, BENCHMARK_VISIT_TIMEOUT,
    DEFAULT_NUM_WORKERS, MIN_WORKERS, MAX_WORKERS,
    BENCHMARK_MAX_TURNS, BENCHMARK_DEFAULT_NUM_RESULTS,
    BENCHMARK_SYSTEM_PROMPT, CAPTURE_FULL_TRACES, SAVE_DETAILED_JSON,
    PROGRESS_UPDATE_FREQUENCY, SHOW_WORKER_DETAILS, SHOW_ETA,
    DEFAULT_OUTPUT_SUFFIX, MAX_REPEATED_RESPONSES,
    MIN_RESPONSE_LENGTH_FOR_REPEAT_CHECK, SAFETY_DELAY_BETWEEN_QUESTIONS,
    METRICS_TO_TRACK, RETRY_FAILED_REQUESTS, LOG_API_ERRORS,
    CONTINUE_ON_WORKER_FAILURE, PORT_ROTATION_ENABLED, PORT_LOCK_TIMEOUT,
    TRACE_TYPES, REPORT_SECTIONS, WORKER_SETTINGS,
    VALIDATE_PORTS_ON_STARTUP, PING_TIMEOUT, DEBUG_MODE,
    VERBOSE_WORKER_OUTPUT, LOG_INCOMPLETE_TOOL_CALLS
)


# Port rotation for load balancing (from config)
PORT_LOCK = threading.Lock()
PORT_COUNTER = 0

def get_next_port():
    """Get next port in round-robin fashion."""
    global PORT_COUNTER
    with PORT_LOCK:
        port = AVAILABLE_PORTS[PORT_COUNTER % len(AVAILABLE_PORTS)]
        PORT_COUNTER += 1
        return port

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

class BenchmarkAgent(SearchAgent):
    """Extended SearchAgent that captures detailed trace information."""
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None, assigned_port: int = None):
        # Use benchmark config by default, fall back to regular config
        if model_config is None:
            model_config = BENCHMARK_MODEL_CONFIG
        
        # Modify config to use assigned port if provided
        if assigned_port:
            config = model_config.copy()
            # Update API endpoints to use assigned port
            self.search_api_url = f'http://{BENCHMARK_BASE_HOST}:{assigned_port}/search'
            self.visit_api_url = f'http://{BENCHMARK_BASE_HOST}:{assigned_port}/visit'
        else:
            config = model_config
            self.search_api_url = None
            self.visit_api_url = None
            
        super().__init__(config)
        self.trace = []
        self.function_calls = []
        self.assigned_port = assigned_port
        
    def reset_trace(self):
        """Reset trace and function call logs."""
        self.trace = []
        self.function_calls = []
        self.reset_conversation()
    
    def build_full_conversation_trace(self, user_query: str) -> List[Dict[str, Any]]:
        """Build complete conversation trace in message format."""
        messages = []
        
        # Add user message
        messages.append({
            "role": "user", 
            "content": user_query
        })
        
        # Process trace items to build conversation
        for item in self.trace:
            if item['type'] == 'model_reasoning':
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": item['content']
                })
            elif item['type'] == 'function_call':
                messages.append({
                    "role": "assistant", 
                    "content": "",
                    "function_call": {
                        "name": item['function'],
                        "arguments": json.dumps(item['arguments'])
                    }
                })
            elif item['type'] == 'function_response':
                messages.append({
                    "role": "function",
                    "name": item.get('function', 'unknown'),
                    "content": item.get('content', '')
                })
            elif item['type'] == 'final_response':
                messages.append({
                    "role": "assistant",
                    "content": item['content']
                })
        
        return messages
    
    def get_clean_final_answer(self) -> str:
        """Extract the final answer without thinking or tool calls."""
        # Find the last final_response in trace
        for item in reversed(self.trace):
            if item['type'] == 'final_response':
                content = item['content']
                # Clean up any thinking tags or artifacts
                if content:
                    # Remove </think> tags and content before them
                    if '</think>' in content:
                        content = content.split('</think>')[-1].strip()
                    # Remove any remaining <think> tags
                    import re
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                return content
        
        # Fallback: look for last model_response
        for item in reversed(self.trace):
            if item['type'] == 'model_response':
                content = item['content']
                if content:
                    # Clean up any thinking tags
                    if '</think>' in content:
                        content = content.split('</think>')[-1].strip()
                    import re
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                return content
        
        return ""
    
    def search(self, user_query: str) -> str:
        """Override search to capture detailed trace with thinking."""
        self.trace = []
        self.function_calls = []
        
        # Initialize conversation with benchmark system prompt
        messages = [{'role': 'system', 'content': BENCHMARK_SYSTEM_PROMPT}]
        messages.append({'role': 'user', 'content': f"{user_query}\n\nBefore using any tools, please first think out loud about what information you need and what your search strategy should be."})
        self.conversation_history = messages.copy()
        
        self.trace.append({
            'type': 'user_query',
            'content': user_query,
            'timestamp': datetime.now().isoformat()
        })
        
        final_response = ""
        turn = 0
        
        # Main search loop
        while True:
            turn += 1
            self.trace.append({
                'type': 'turn_start',
                'turn': turn,
                'timestamp': datetime.now().isoformat()
            })
            
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
            
            # Extract thinking from <think> tags if present
            thinking_text = None
            if regular_content and '<think>' in regular_content and '</think>' in regular_content:
                think_match = re.search(r'<think>(.*?)</think>', regular_content, re.DOTALL)
                if think_match:
                    thinking_text = think_match.group(1).strip()
                    regular_content = regular_content.replace(think_match.group(0), '').strip()
            
            # Log reasoning/thinking
            if reasoning_content and reasoning_content.strip():
                self.trace.append({
                    'type': 'model_reasoning',
                    'content': reasoning_content,
                    'timestamp': datetime.now().isoformat()
                })
            elif thinking_text:
                self.trace.append({
                    'type': 'model_thinking',
                    'content': thinking_text,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Log regular content if available
            if regular_content and regular_content.strip():
                self.trace.append({
                    'type': 'model_response',
                    'content': regular_content,
                    'timestamp': datetime.now().isoformat()
                })
            
            messages.append(last_response)
            
            # Check for function call (handle both standard and nous formats)
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
                # First try complete tool call
                tool_call_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL)
                if tool_call_match:
                    try:
                        tool_call_data = json.loads(tool_call_match.group(1))
                        function_name = tool_call_data.get('name')
                        function_args = tool_call_data.get('arguments', {})
                        function_call_detected = True
                    except json.JSONDecodeError:
                        pass
                # If no complete tool call, check for incomplete/truncated tool call
                elif '<tool_call>' in content and '</tool_call>' not in content:
                    print(f"[DEBUG] Incomplete tool call detected, continuing to next turn")
                    # Don't break, continue to next turn to let model complete the call
                    pass
            
            if function_call_detected and function_name and function_args:
                function_call_info = {
                    'function': function_name,
                    'arguments': function_args,
                    'timestamp': datetime.now().isoformat()
                }
                self.function_calls.append(function_call_info)
                
                self.trace.append({
                    'type': 'function_call',
                    'function': function_name,
                    'arguments': function_args,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Execute function
                function_to_call = self.available_functions[function_name]
                if function_name == 'web_search':
                    # Use assigned port if available
                    if self.search_api_url:
                        original_url = self.search_api_url
                        # Temporarily override the agent's search URL
                        function_response = self._web_search_with_port(
                            query=function_args.get('query'),
                            number_results=function_args.get('number_results', 1),
                            api_url=self.search_api_url
                        )
                    else:
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
                    # Use assigned port if available
                    if self.visit_api_url:
                        function_response = self._web_visit_with_port(
                            url=function_args.get('url'),
                            api_url=self.visit_api_url
                        )
                    else:
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
                
                messages.append({
                    'role': 'function',
                    'name': function_name,
                    'content': formatted_response
                })

                self.trace.append({
                    'type': 'function_response',
                    'function': function_name,
                    'content': formatted_response,
                    'response_length': len(formatted_response),
                    'response_preview': formatted_response,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                # Check if this looks like reasoning/planning without immediate tool call
                response_content = last_response.get('content', '') if isinstance(last_response, dict) else str(last_response)
                
                # If response mentions searching, planning, or strategy but no tool call, continue
                planning_indicators = ['search for', 'will search', 'strategy', 'plan to', 'need to find', 'look for']
                incomplete_tool_call = '<tool_call>' in response_content and '</tool_call>' not in response_content
                
                if any(indicator in response_content.lower() for indicator in planning_indicators) or incomplete_tool_call:
                    print(f"[DEBUG] Turn {turn}: Model is planning/reasoning, continuing...")
                    # Don't break, continue conversation
                    pass
                else:
                    # Model provided final response
                    if isinstance(last_response, dict) and 'content' in last_response:
                        final_response = last_response['content']
                    elif isinstance(last_response, str):
                        final_response = last_response
                    
                    self.trace.append({
                        'type': 'final_response',
                        'content': final_response,
                        'timestamp': datetime.now().isoformat()
                    })
                    break
            
            # Safety check
            if turn > BENCHMARK_MAX_TURNS:
                self.trace.append({
                    'type': 'max_turns_reached',
                    'timestamp': datetime.now().isoformat()
                })
                break
                
            # Check for repeated responses
            if len(messages) >= 4:
                last_content = messages[-1].get('content', '') if isinstance(messages[-1], dict) else ''
                prev_content = messages[-3].get('content', '') if isinstance(messages[-3], dict) else ''
                if last_content and last_content == prev_content and len(last_content) > 100:
                    self.trace.append({
                        'type': 'repeated_response_detected',
                        'timestamp': datetime.now().isoformat()
                    })
                    break
        
        self.conversation_history = messages
        return final_response
    
    def _web_search_with_port(self, query: str, number_results: int = 3, api_url: str = None):
        """Web search using specific API port."""
        url = api_url or f'http://{BENCHMARK_BASE_HOST}:{get_next_port()}/search'
        payload = {
            'query': query,
            'number_results': number_results
        }
        
        try:
            response = requests.post(url, json=payload, timeout=BENCHMARK_SEARCH_TIMEOUT)
            response.raise_for_status()
            return json.dumps(response.json())
        except Exception as e:
            return json.dumps({'error': f"Search error: {str(e)}", 'results': []})
    
    def _web_visit_with_port(self, url: str, api_url: str = None):
        """Web visit using specific API port."""
        visit_url = api_url or f'http://{BENCHMARK_BASE_HOST}:{get_next_port()}/visit'
        payload = {'url': url}
        
        try:
            response = requests.post(visit_url, json=payload, timeout=BENCHMARK_VISIT_TIMEOUT)
            response.raise_for_status()
            import json
            return json.dumps(response.json())
        except Exception as e:
            import json
            return json.dumps({'error': f"Visit error: {str(e)}", 'content': ''})


def process_question_worker(args):
    """Worker function to process a single question with assigned port."""
    question_data, assigned_port, worker_id = args
    
    try:
        question = question_data.get('question')
        question_id = question_data.get('question_id', question_data.get('id', f'q{worker_id}'))
        
        # Initialize agent with assigned port
        agent = BenchmarkAgent(assigned_port=assigned_port)
        agent.reset_trace()
        start_time = time.time()
        
        final_answer = agent.search(question)
        response_time = time.time() - start_time
        
        search_calls = sum(1 for c in agent.function_calls if c['function'] == 'web_search')
        visit_calls = sum(1 for c in agent.function_calls if c['function'] == 'web_visit')
        
        
        # Count thinking and reasoning entries
        thinking_entries = sum(1 for t in agent.trace if t['type'] in ['model_thinking', 'model_reasoning', 'model_planning'])
        
        # Get clean final answer and build full conversation trace
        clean_final_answer = agent.get_clean_final_answer()
        full_trace = agent.build_full_conversation_trace(question)
        
        result = {
            'question_id': question_id,
            'question': question,
            'final_answer': clean_final_answer,
            'full_trace': full_trace,
            'response_time': response_time,
            'num_search_calls': search_calls,
            'num_visit_calls': visit_calls,
            'total_function_calls': len(agent.function_calls),
            'thinking_entries': thinking_entries,
            'total_turns': len([t for t in agent.trace if t['type'] == 'turn_start']),
            'success': True,
            'error': None,
            'search_history': agent.search_history,
            'worker_id': worker_id,
            'assigned_port': assigned_port
        }
        
        print(f"[Worker {worker_id}:{assigned_port}] ✓ {question_id} in {response_time:.2f}s (search: {search_calls}, visit: {visit_calls})")
        return result
        
    except Exception as e:
        response_time = time.time() - start_time if 'start_time' in locals() else 0
        result = {
            'question_id': question_data.get('question_id', question_data.get('id', f'q{worker_id}')),
            'question': question_data.get('question'),
            'final_answer': None,
            'full_trace': [{"role": "user", "content": question_data.get('question')}],
            'response_time': response_time,
            'num_search_calls': 0,
            'num_visit_calls': 0,
            'total_function_calls': 0,
            'thinking_entries': 0,
            'total_turns': 0,
            'success': False,
            'error': str(e),
            'search_history': [],
            'worker_id': worker_id,
            'assigned_port': assigned_port
        }
        print(f"[Worker {worker_id}:{assigned_port}] ✗ {question_data.get('question_id', 'unknown')}: {str(e)}")
        return result

def run_benchmark(questions_file: str, output_file: str = None, save_traces: bool = CAPTURE_FULL_TRACES, num_workers: int = DEFAULT_NUM_WORKERS):
    """Run benchmark on questions from file and generate comprehensive report."""
    print(f"\n{'='*80}")
    print(f"Search Agent Benchmark")
    print(f"{'='*80}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load questions
    if questions_file.endswith('.csv'):
        questions_df = pd.read_csv(questions_file)
        if 'question' not in questions_df.columns:
            raise ValueError("CSV must have a 'question' column")
        questions = questions_df.to_dict('records')
    elif questions_file.endswith('.json'):
        with open(questions_file, 'r') as f:
            questions = json.load(f)
        if isinstance(questions[0], str):
            questions = [{'question': q, 'question_id': f'q{i+1}'} for i, q in enumerate(questions)]
    else:
        raise ValueError("Questions file must be CSV or JSON")
    
    print(f"Loaded {len(questions)} questions from: {questions_file}")
    
    print(f"Using {num_workers} workers across ports: {AVAILABLE_PORTS}")
    
    # Prepare worker arguments with port assignments
    worker_args = []
    for i, question_data in enumerate(questions):
        # Assign port in round-robin fashion
        assigned_port = AVAILABLE_PORTS[i % len(AVAILABLE_PORTS)]
        # Add question_id if missing
        if 'question_id' not in question_data and 'id' not in question_data:
            question_data['question_id'] = f'q{i+1}'
        worker_args.append((question_data, assigned_port, i+1))
    
    # Process questions in parallel
    start_time = time.time()
    results = []
    
    print(f"\nProcessing {len(questions)} questions with {num_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_args = {executor.submit(process_question_worker, args): args for args in worker_args}
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_args):
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                # Progress update
                elapsed = time.time() - start_time
                avg_time = elapsed / completed
                remaining = len(questions) - completed
                eta = avg_time * remaining
                
                # Show progress updates intelligently
                show_progress = (
                    completed <= 3 or  # First few
                    completed % max(1, len(questions) // 10) == 0 or  # Every 10%
                    completed == len(questions) or  # Final
                    (elapsed > 30 and completed % 5 == 0)  # Every 5 if taking long
                )
                if show_progress:
                    success_rate = sum(1 for r in results if r['success']) / len(results) * 100
                    print(f"Progress: {completed}/{len(questions)} ({completed/len(questions)*100:.1f}%) - "
                          f"Success: {success_rate:.1f}% - Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s")
                      
            except Exception as e:
                args = future_to_args[future]
                question_data, assigned_port, worker_id = args
                print(f"[Worker {worker_id}:{assigned_port}] ✗ Exception: {str(e)}")
                # Add a failed result
                results.append({
                    'question_id': question_data.get('question_id', f'q{worker_id}'),
                    'question': question_data.get('question'),
                    'final_answer': None,
                    'full_trace': [{"role": "user", "content": question_data.get('question')}],
                    'response_time': 0,
                    'num_search_calls': 0,
                    'num_visit_calls': 0,
                    'total_function_calls': 0,
                    'thinking_entries': 0,
                    'total_turns': 0,
                    'success': False,
                    'error': f"Worker exception: {str(e)}",
                    'search_history': [],
                    'worker_id': worker_id,
                    'assigned_port': assigned_port
                })
    
    total_time = time.time() - start_time
    print(f"\n🎉 Completed {len(questions)} questions in {total_time:.2f}s ({total_time/len(questions):.2f}s avg)")
    
    # Generate and save reports
    report = generate_report(results)
    
    if output_file:
        # Save text report
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_file}")
        
        # Save detailed JSON results if traces are included
        if save_traces:
            json_file = output_file.replace('.txt', DEFAULT_OUTPUT_SUFFIX)
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Detailed results saved to: {json_file}")
    else:
        print(report)
    
    return results


def format_trace_as_messages(trace: List[Dict[str, Any]]) -> str:
    """Format trace entries as message objects similar to the search agent output."""
    output = []
    
    for item in trace:
        if item['type'] == 'model_reasoning':
            content = item['content']
            output.append(f"{{'role': 'assistant', 'content': '', 'reasoning_content': '''\n{content}\n'''}}")
        elif item['type'] == 'function_call':
            args_json = json.dumps(item['arguments'])
            output.append(f"{{'role': 'assistant', 'content': '', 'function_call': {{'name': '{item['function']}', 'arguments': '{args_json}'}}}},")
        elif item['type'] == 'function_response':
            response_preview = item.get('content', '')
            output.append(f"{{'role': 'function', 'name': '{item.get('function', 'unknown')}', 'content': '{response_preview}'}},")
        elif item['type'] == 'final_response':
            content = item['content']
            output.append(f"{{'role': 'assistant', 'content': '''\n{content}\n'''}}")
    
    return "\n\n".join(output)


def generate_report(results: List[Dict[str, Any]]) -> str:
    """Generate a comprehensive benchmark report."""
    report = StringIO()
    
    # Header
    report.write(f"\n{'='*80}\n")
    report.write(f"SEARCH AGENT BENCHMARK REPORT\n")
    report.write(f"{'='*80}\n")
    report.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Summary statistics
    total = len(results)
    successful = sum(1 for r in results if r['success'])
    failed = total - successful
    
    report.write("SUMMARY\n")
    report.write("-" * 40 + "\n")
    report.write(f"Total questions: {total}\n")
    report.write(f"Successful: {successful} ({successful/total*100:.1f}%)\n")
    report.write(f"Failed: {failed} ({failed/total*100:.1f}%)\n")
    
    if successful > 0:
        success_results = [r for r in results if r['success']]
        avg_time = sum(r['response_time'] for r in success_results) / len(success_results)
        avg_search = sum(r['num_search_calls'] for r in success_results) / len(success_results)
        avg_visit = sum(r['num_visit_calls'] for r in success_results) / len(success_results)
        avg_thinking = sum(r['thinking_entries'] for r in success_results) / len(success_results)
        avg_turns = sum(r['total_turns'] for r in success_results) / len(success_results)
        
        report.write(f"\nPerformance Metrics:\n")
        report.write(f"Average response time: {avg_time:.2f}s\n")
        report.write(f"Average search calls: {avg_search:.2f}\n")
        report.write(f"Average visit calls: {avg_visit:.2f}\n")
        report.write(f"Average thinking entries: {avg_thinking:.2f}\n")
        report.write(f"Average turns: {avg_turns:.2f}\n")
    
    # Results table
    report.write(f"\n\nRESULTS TABLE\n")
    report.write("-" * 100 + "\n")
    report.write(f"{'ID':<10} {'Time':<8} {'Search':<8} {'Visit':<8} {'Think':<8} {'Turns':<8} {'Status':<10} {'Question':<30}\n")
    report.write("-" * 100 + "\n")
    
    for r in results:
        status = "Success" if r['success'] else "Failed"
        question_short = r['question']
        report.write(f"{r['question_id']:<10} {r['response_time']:<8.2f} {r['num_search_calls']:<8} "
                    f"{r['num_visit_calls']:<8} {r['thinking_entries']:<8} {r['total_turns']:<8} "
                    f"{status:<10} {question_short:<30}\n")
    
    # Detailed results
    report.write(f"\n\nDETAILED RESULTS\n")
    report.write("=" * 100 + "\n")
    
    for r in results:
        report.write(f"\nQuestion ID: {r['question_id']}\n")
        report.write(f"Question: {r['question']}\n")
        report.write(f"Status: {'Success' if r['success'] else 'Failed'}\n")
        report.write(f"Response Time: {r['response_time']:.2f}s\n")
        report.write(f"Function Calls: {r['total_function_calls']} (search: {r['num_search_calls']}, visit: {r['num_visit_calls']})\n")
        report.write(f"Thinking Entries: {r['thinking_entries']}\n")
        report.write(f"Total Turns: {r['total_turns']}\n")
        
        if r['final_answer']:
            report.write(f"\nFinal Answer:\n")
            report.write("-" * 50 + "\n")
            answer = r['final_answer']
            report.write(answer + "\n")
        
        if r['error']:
            report.write(f"\nError: {r['error']}\n")
        
        # Full trace in message format (from full_trace field)
        if r.get('full_trace'):
            report.write(f"\nFull Conversation Trace:\n")
            report.write("-" * 50 + "\n")
            for msg in r['full_trace']:
                if msg['role'] == 'user':
                    report.write(f"User: {msg['content']}\n\n")
                elif msg['role'] == 'assistant' and msg.get('reasoning_content'):
                    reasoning = msg['reasoning_content']
                    report.write(f"Reasoning: {reasoning}\n\n")
                elif msg['role'] == 'assistant' and msg.get('function_call'):
                    report.write(f"Tool Call: {msg['function_call']['name']}({msg['function_call']['arguments']})\n\n")
                elif msg['role'] == 'function':
                    response = msg['content']
                    report.write(f"Tool Response: {response}\n\n")
                elif msg['role'] == 'assistant' and msg.get('content'):
                    content = msg['content']
                    report.write(f"Assistant: {content}\n\n")
        
        report.write("-" * 100 + "\n")
    
    return report.getvalue()


def main():
    parser = argparse.ArgumentParser(description='Benchmark the Search Agent with full trace capture and multiprocessing')
    parser.add_argument('questions_file', help='Path to questions file (CSV or JSON)')
    parser.add_argument('-o', '--output', help='Output file for report (optional)')
    parser.add_argument('--no-traces', action='store_true', help='Disable trace capture to save space')
    parser.add_argument('-w', '--workers', type=int, default=DEFAULT_NUM_WORKERS, 
                       help=f'Number of worker processes (default: {DEFAULT_NUM_WORKERS}, range: {MIN_WORKERS}-{MAX_WORKERS})')
    parser.add_argument('--validate-ports', action='store_true',
                       help='Validate API endpoints before starting (adds startup delay)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show detailed worker output')
    
    args = parser.parse_args()
    
    # Validate worker count
    if args.workers < MIN_WORKERS or args.workers > MAX_WORKERS:
        print(f"Warning: Worker count {args.workers} outside valid range ({MIN_WORKERS}-{MAX_WORKERS}). Clamping to valid range.")
        args.workers = max(MIN_WORKERS, min(MAX_WORKERS, args.workers))
    
    run_benchmark(args.questions_file, args.output, save_traces=not args.no_traces, num_workers=args.workers)


if __name__ == '__main__':
    main()
import json
import time
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import argparse
from io import StringIO

from search_agent import SearchAgent
from config import MODEL_CONFIG


class BenchmarkAgent(SearchAgent):
    """Extended SearchAgent that captures detailed trace information."""
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        super().__init__(model_config)
        self.trace = []
        self.function_calls = []
        
    def reset_trace(self):
        """Reset trace and function call logs."""
        self.trace = []
        self.function_calls = []
        self.reset_conversation()
    
    def search(self, user_query: str, max_turns: int = 5) -> str:
        """Override search to capture detailed trace."""
        self.trace = []
        self.function_calls = []
        
        messages = self._add_system_prompt()
        messages.append({'role': 'user', 'content': user_query})
        self.conversation_history = messages.copy()
        
        self.trace.append({
            'type': 'user_query',
            'content': user_query,
            'timestamp': datetime.now().isoformat()
        })
        
        final_response = ""
        
        for turn in range(max_turns):
            self.trace.append({
                'type': 'turn_start',
                'turn': turn + 1,
                'timestamp': datetime.now().isoformat()
            })
            
            responses = []
            for response in self.llm.chat(
                messages=messages,
                functions=self.functions,
                stream=True
            ):
                responses.append(response)
            
            if not responses:
                break
                
            last_response = responses[-1]
            messages.append(last_response)
            
            if last_response.get('function_call'):
                function_name = last_response['function_call']['name']
                function_args = json.loads(last_response['function_call']['arguments'])
                
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
                
                function_to_call = self.available_functions[function_name]
                if function_name == 'web_search':
                    function_response = function_to_call(
                        query=function_args.get('query'),
                        number_results=function_args.get('number_results', 1)
                    )
                elif function_name == 'web_visit':
                    function_response = function_to_call(
                        url=function_args.get('url')
                    )
                
                self.trace.append({
                    'type': 'function_response',
                    'function': function_name,
                    'response': function_response[:500] + '...' if len(function_response) > 500 else function_response,
                    'timestamp': datetime.now().isoformat()
                })
                
                messages.append({
                    'role': 'function',
                    'name': function_name,
                    'content': function_response
                })
            else:
                if 'content' in last_response:
                    final_response = last_response['content']
                    
                    self.trace.append({
                        'type': 'agent_response',
                        'content': final_response,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    if any(phrase in final_response.lower() for phrase in 
                           ['based on my search', 'in summary', 'to conclude', 
                            'the answer is', 'according to the information']):
                        break
        
        self.conversation_history = messages
        return final_response


def run_benchmark(questions_file: str, output_file: str = None):
    """Run benchmark on questions from file and generate report."""
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
    
    # Initialize agent
    agent = BenchmarkAgent()
    results = []
    
    # Process each question
    for i, item in enumerate(questions):
        question = item.get('question')
        question_id = item.get('question_id', item.get('id', f'q{i+1}'))
        
        print(f"\n[{i+1}/{len(questions)}] Processing: {question[:80]}...")
        
        agent.reset_trace()
        start_time = time.time()
        
        try:
            final_answer = agent.search(question)
            response_time = time.time() - start_time
            
            search_calls = sum(1 for c in agent.function_calls if c['function'] == 'web_search')
            visit_calls = sum(1 for c in agent.function_calls if c['function'] == 'web_visit')
            
            result = {
                'question_id': question_id,
                'question': question,
                'final_answer': final_answer,
                'response_time': response_time,
                'num_search_calls': search_calls,
                'num_visit_calls': visit_calls,
                'total_function_calls': len(agent.function_calls),
                'success': True,
                'error': None,
                'trace': agent.trace
            }
            
            print(f"  ✓ Success in {response_time:.2f}s (search: {search_calls}, visit: {visit_calls})")
            
        except Exception as e:
            response_time = time.time() - start_time
            result = {
                'question_id': question_id,
                'question': question,
                'final_answer': None,
                'response_time': response_time,
                'num_search_calls': 0,
                'num_visit_calls': 0,
                'total_function_calls': 0,
                'success': False,
                'error': str(e),
                'trace': agent.trace
            }
            print(f"  ✗ Failed: {str(e)}")
        
        results.append(result)
        time.sleep(1)  # Rate limiting
    
    # Generate report
    report = generate_report(results)
    
    # Save or print report
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_file}")
    else:
        print(report)
    
    return results


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
        
        report.write(f"\nAverage response time: {avg_time:.2f}s\n")
        report.write(f"Average search calls: {avg_search:.2f}\n")
        report.write(f"Average visit calls: {avg_visit:.2f}\n")
    
    # Results table
    report.write(f"\n\nRESULTS TABLE\n")
    report.write("-" * 80 + "\n")
    report.write(f"{'ID':<10} {'Time':<8} {'Search':<8} {'Visit':<8} {'Status':<10} {'Question':<30}\n")
    report.write("-" * 80 + "\n")
    
    for r in results:
        status = "Success" if r['success'] else "Failed"
        question_short = r['question'][:30] + "..." if len(r['question']) > 30 else r['question']
        report.write(f"{r['question_id']:<10} {r['response_time']:<8.2f} {r['num_search_calls']:<8} "
                    f"{r['num_visit_calls']:<8} {status:<10} {question_short:<30}\n")
    
    # Detailed results
    report.write(f"\n\nDETAILED RESULTS\n")
    report.write("=" * 80 + "\n")
    
    for r in results:
        report.write(f"\nQuestion ID: {r['question_id']}\n")
        report.write(f"Question: {r['question']}\n")
        report.write(f"Status: {'Success' if r['success'] else 'Failed'}\n")
        report.write(f"Response Time: {r['response_time']:.2f}s\n")
        report.write(f"Function Calls: {r['total_function_calls']} (search: {r['num_search_calls']}, visit: {r['num_visit_calls']})\n")
        
        if r['final_answer']:
            report.write(f"\nFinal Answer:\n")
            report.write("-" * 40 + "\n")
            report.write(r['final_answer'][:500] + "...\n" if len(r['final_answer']) > 500 else r['final_answer'] + "\n")
        
        if r['error']:
            report.write(f"\nError: {r['error']}\n")
        
        # Trace summary
        report.write(f"\nTrace Summary:\n")
        report.write("-" * 40 + "\n")
        for item in r['trace']:
            if item['type'] == 'function_call':
                report.write(f"  → {item['function']}({item['arguments']})\n")
            elif item['type'] == 'agent_response':
                report.write(f"  ← Agent: {item['content'][:100]}...\n")
        
        report.write("-" * 80 + "\n")
    
    return report.getvalue()


def main():
    parser = argparse.ArgumentParser(description='Benchmark the Search Agent')
    parser.add_argument('questions_file', help='Path to questions file (CSV or JSON)')
    parser.add_argument('-o', '--output', help='Output file for report (optional)')
    
    args = parser.parse_args()
    
    run_benchmark(args.questions_file, args.output)


if __name__ == '__main__':
    main()
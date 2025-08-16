import os
import json
import glob
import logging
import argparse
import time
import random
import threading
from concurrent.futures import ProcessPoolExecutor
import openai
import multiprocessing
import hashlib
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("worker.log"), logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# Create a semaphore to limit concurrent API calls
# We use a multiprocessing semaphore since we're using ProcessPoolExecutor
api_semaphore = None  # Will be initialized in main()


import os, sys, json
import requests
import openai
from logging import getLogger
import time
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
logger = getLogger("sglang")
from tools import WebSearchTool, VisitWebpageTool, execute_code
from code_exe import PythonExecutor
MODEL_NAME="/home/slurm/tuenv2/tuenv/model_hub/qwen3/Qwen3-235B-A22B"
URL_API="http://192.168.0.8:1235/v1"
IS_QWEN3=True
# tokenizer = AutoTokenizer.from_pretrained("/home/slurm/tuenv2/tuenv/model_hub/qwen3/Qwen3-235B-A22B")
tokenizer=None
def apply_chat_template(messages, is_qwen3=IS_QWEN3, tools=None):
    if is_qwen3 == False:
        messages = [
            {
                "role": i['role'],
                "content": [{"text": i['content']}]
            }
            for i in messages
        ]
    # print(messages)
    return tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True, tools=tools).replace(
            "<｜Assistant｜><｜Assistant｜>",
            "<｜Assistant｜><think>"
        )

import os
from typing import Any, Dict, List

from datetime import datetime, timezone
from pydantic import BaseModel, Field



class ConfigConstants:
    """Constants used across configuration."""

    # Template fragments
    THINK_TAG_OPEN = "<think>"
    THINK_TAG_CLOSE = "</think>"
    TOOL_RESPONSE_OPEN = "<tool_response>"
    TOOL_RESPONSE_CLOSE = "</tool_response>"
    CODE_BLOCK_START = "```python"
    CODE_BLOCK_END = "```"
    END_CODE = "<end_code>"
    INSTRUCTIONS_OPEN = "<instructions>"
    INSTRUCTIONS_CLOSE = "</instructions>"

    TOOL_CALL_EXAMPLE = (f"{CODE_BLOCK_START}\n"
                         'web_search(queries=["# the query to search", ...]) or '
                         'page_visit(urls=["list of urls to visit", ...])\n'
                         f"{CODE_BLOCK_END}{END_CODE}")

    # Stop sequences
    DEFAULT_STOP_SEQUENCE = [END_CODE]

    # Templates
    DUPLICATE_QUERY_TEMPLATE = "I have already searched for this query: {query}. Please don't search for it again."
    DUPLICATE_URL_TEMPLATE = ("I have already visited this url: {url}. Please don't visit it again.")
    SEARCH_SUFFIX = ("This results may not enough to provide useful information. "
                     "I must do more research or use page_visit tool to get detailed information. \n")
    PAGE_VISIT_SUFFIX = ("I have just got some new information. Maybe it's helpful but let me see if it "
                         "contains something interesting.\n"
                         "I should note the interesting key ideas/ exact quote along with citations so that "
                         "I can use it in the final answer.\n"
                         "I can not provider the final answer when I don't have enough information or when "
                         "I am not sure about the answer.\n")


class ToolConfig(BaseModel):
    """Configuration for tools."""

    max_search_results: int = 4
    max_search_queries: int = 2
    max_urls_to_visit: int = 3


class LLMConfig(BaseModel):
    """Configuration for LLM clients."""

    model: str = Field(default_factory=lambda: os.getenv("R_MODEL", "r1"))
    temperature: float = Field(default_factory=lambda: float(os.getenv("R_TEMPERATURE", "0.2")))
    top_p: float = 0.95
    presence_penalty: float = Field(default_factory=lambda: float(os.getenv("R_PRESENCE_PENALTY", "0")))
    stop_sequence: List[str] = Field(default_factory=lambda: ConfigConstants.DEFAULT_STOP_SEQUENCE)
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "empty"))
    base_url: str = Field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "http://localhost:4000"))
    report_model: str = Field(default_factory=lambda: os.getenv("R_REPORT_MODEL", "gpt-4o"))

    def get_effective_stop_sequence(self, trace_has_turns: bool = False) -> List[str]:
        """Get effective stop sequence based on state."""
        if trace_has_turns:
            # Include </think> tag as a stop token when we have turns
            return list(set(self.stop_sequence).union([ConfigConstants.THINK_TAG_CLOSE]))
        return self.stop_sequence


def check_for_repeated_sections(text: str, repetition_threshold: int = 1) -> tuple[bool, list[str]]:
    """
    Checks if any paragraph-like section in a given text is repeated more than a specified threshold.

    A paragraph is considered a block of text separated by one or more newline characters.
    The function first cleans up the text by stripping leading/trailing whitespace
    from each line and removing empty lines.

    Args:
        text: The input string to check for repeated sections.
        repetition_threshold: The minimum number of times a section must be repeated
                              to be considered a repetition. Defaults to 1 (meaning
                              it appears more than once).

    Returns:
        A tuple containing:
        - A boolean value: True if a repeated section is found, False otherwise.
        - A list of the unique paragraphs that were found to be repeated.
    """
    # Split the text into paragraphs based on one or more newline characters.
    # We filter out any empty strings that might result from multiple newlines.
    paragraphs = [p.strip() for p in text.strip().split('\n') if p.strip()]

    if not paragraphs:
        return False, []

    # Count the occurrences of each unique paragraph.
    paragraph_counts = {}
    for p in paragraphs:
        if len(p.split())>=20:
            paragraph_counts[p] = paragraph_counts.get(p, 0) + 1

    # Find paragraphs that are repeated more than the threshold.
    repeated_paragraphs = [
        p for p, count in paragraph_counts.items() if count > 2
    ]

    # Determine if any repetitions were found.
    has_repetitions = len(repeated_paragraphs) > 0

    return has_repetitions, repeated_paragraphs

class AgentConfig(BaseModel):
    """Configuration for the agent."""

    tool: ToolConfig = ToolConfig()
    llm: LLMConfig = LLMConfig()
    
    system_prompt: str = f"""
You are II Researcher, developed by Intelligent Internet.
You first thinks about the reasoning process in the mind and then provides the user with the answer. 
You are specialized in multistep reasoning.
<intro>
You excel at the following tasks:
1. High-level information gathering, research, fact-checking, and structured documentation.
2. Skillful construction of efficient, targeted queries to extract the most relevant information. (SEARCH SKILL)
3. Applying programming to analyze, transform, or solve complex tasks far beyond basic software development.
</intro>

<system_capability>
- You first think about the reasoning process in the mind and then provide the user with the answer.
- You are trained for deep, multi-step inference using prior knowledge, training data, and live searches.
- You write and execute Python code for advanced data collection (via web_search) and content retrieval (via visit_webpage).
</system_capability>

<reasoning_process>
When you need to solve a problem or answer a question:
1. Start by thinking through the problem logically step by step.
2. Use web_search for initial discovery. Use visit_webpage to confirm, deepen, and validate.
3. Adapt and extend reasoning as new information emerges.
4. Do not finalize answers until you are confident and supported by evidence.
5. Only use </think> when all reasoning and code execution is complete.
6. After </think>, provide only the final answer—clear, accurate, and with source citations if appropriate.
</reasoning_process>

<critical_tool_calling_format>
CRITICAL: You MUST call python code using ONLY this exact format. DO NOT write any other code blocks!

To call a python, write EXACTLY like this:
<start_code>
```python
...
```
<end_code>

Examples of CORRECT tool calls:
<start_code>
```python
response_str = web_search(query="# the query to search")
print(response_str)

```<end_code>

<start_code>
```python
response_str_visit = visit_webpage(url="the url must be returned by web_search")
print(response_str_visit)
```<end_code>



WRONG - DO NOT DO THIS:
- Do not forget the <start_code> and <end_code>  tag
- Do not use any format other than: <start_code>```python\n(...)\\n```<end_code>

</critical_tool_calling_format>

<available_functions>
You have access to these function:

{{tool}}

</available_functions>



<rules>
1. **Code Usage**: ALWAYS use the code calling format shown above
2. **Information Priority**: authoritative data > web search > model's internal knowledge
3. **Search Strategy**: 
   - Think first, then search.
   - Split complex questions into multiple subqueries.
   - Compare and cross-verify sources when needed.
   - Try visit multiple URLs from search results for for context and validation.
4. **Research Requirements**:
   - Do not make assumptions - verify with actions
   - All claims MUST be supported by search results
   - Research THOROUGHLY before providing final answer
   - Final answer should be detailed with citations
   - Be exhaustive and cautious—missing or unverified data = incomplete task.
</rules>

<error_handling>
- When errors occur, first verify function names and arguments
- Attempt to fix issues based on error messages
- Try alternative methods if first approach fails
- Report failure reasons to user and request assistance
</error_handling>

<final_reminders>
REMEMBER:
1. ONLY use the code calling format: <start_code>```python\n...\n```<end_code>
2. All code calls MUST happen before </think> tag
3. Use all available tools to confirm and justify reasoning.
4. If unsure—search, verify, confirm.
5. Guessing without checking = disqualification.
6. After </think>, only provide the final answer
7. ALWAYS END THE CODE CALLING WITH <end_code> DONT FORGET THIS, THIS IS VERY IMPORTANT, IF NOT YOU ARE STUPID.
</final_reminders>

<very_important_rule>
- All the python calls MUST happen before the </think> tag. Only use </think> tag when you are sure about the answer.
- Research DETAILS and THOROUGHLY about the topic before you provide the final answer.
- After the </think> tag, You can only provide the final answer.
- Don't rely only on your reasoning process, when you confused, you can perform an action to get more information.
- Do not make any assumptions, if you are not sure about the answer, you can perform an action to get more information.
- All the information you claim MUST be supported by the search results if it's come from your reasoning process. Perform action to confirm your claims.
- When you are not sure about the answer, You don't make a guess.
- Every part of the answer should be supported by the search results.
- The Search function maybe only return the relevant part not directly answer your query, you can try to visit the page and see more information.
    For example: 
        Input:    web_search(query=\"creator of Archer and Sealab 2021\", 10)
        Output: Title: Archer film
        Preview: The thirteenth season of the animated television series Archer ... 
        Reasoning: Maybe inside the Archer Film we can find the creator of Archer and Sealab 2021, maybe some information in there because we maybe find the creator of Archer film information inside this page.
- You only have access to the URL come from the web_search output.
- You should try many steps as much as possible to confirm your answer and your reasoning.
</final_important_rule>
Current date: {{current_date}}

Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
"""

    instructions: str = f"""
I'm remembering that the user asked me to follow these instructions:
IMPORTANT: 
    - All the function calls MUST happen before the </think> tag. Only use </think> tag when you are sure about the answer.
    - Research DETAILS and THOROUGHLY about the topic before you provide the final answer.
    - After the </think> tag, You can only provide the final answer.
    - Don't rely only on your reasoning process, when you confused, you can perform an action to get more information.
    - Do not make any assumptions, if you are not sure about the answer, you can perform an action to get more information.
    - All the information you claim MUST be supported by the search results if it's come from your reasoning process. Perform action to confirm your claims.
    - When you are not sure about the answer, You don't make a guess.
    - Every part of the answer should be supported by the search results.
    - The Search function maybe only return the relevant part not directly answer your query, you can try to visit the page and see more information.
        For example: 
            Input:    web_search(query=\"creator of Archer and Sealab 2021\", 10)
            Output: Title: Archer film
            Preview: The thirteenth season of the animated television series Archer ... 
            Reasoning: Maybe inside the Archer Film we can find the creator of Archer and Sealab 2021, maybe some information in there because we maybe find the creator of Archer film information inside this page.
    - You only have access to the URL come from the web_search output.
    - You should try many steps as much as possible to confirm your answer and your reasoning.
    - Never stop until you find the source confirm.
The user mentioned that I need to use the information obtained from visit_webpage, not only rely on the snippet preview text from web_search. Another important point is that I should think deeply and break down the problem into multiple steps to find the necessary information.
Sometimes, the search function may only return relevant parts, not a direct answer to my query. However, based on the snippets, I can identify some promising pages that might contain the information I need.
After several failed attempts, I should think outside the box and come up with a new strategy to answer the question. My task is to find as many sources and as much information as possible through visit_webpage to confirm my answer.
Okay, let’s think step by step to solve the question by myself first. This will help me create a clear plan to solve the problem through searching and visiting pages afterward.
"""
        # - Don't worry about the resources, You must answer the question with infinity money/power machine.
# 
    # Use constants for repeated strings
    duplicate_query_template: str = ConfigConstants.DUPLICATE_QUERY_TEMPLATE
    duplicate_url_template: str = ConfigConstants.DUPLICATE_URL_TEMPLATE
# from loguru import logger
def convert_timestamp(timestamp):
    """Convert a timestamp to a human-readable format."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
def call_api(prompt, model, base_url=None, max_retries=3, temperature=0.6, max_tokens=32768, top_p=0.95, top_k=20, other_kwargs={}):
    """Call the OpenAI API with retries."""
    # Use the semaphore to limit concurrent API calls
    if True:
        logger.info(f"Acquired API semaphore, making API call")
        client_kwargs = {"api_key": "EMPTY"}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        client = openai.Client(**client_kwargs)
        
        retries = 0
        retry_delay = 1
        # messages=[{"role": "user", "content": prompt}]
        # if isinsta
        while retries < max_retries:
            try:
                response = client.completions.create(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": True},
                    },
                    timeout=10000000,
                    **other_kwargs
                )
                # print(response.choices[0])
                return {
                    "response": response.choices[0].text,
                    "response_information": {
                        "total_tokens": response.usage.total_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "is_complete": response.choices[0].finish_reason == "stop",
                        "is_stop_by_length": response.choices[0].finish_reason =="length",
                        "matched_stop": getattr(response.choices[0] , "stop_reason", getattr(response.choices[0] , "matched_stop", None)) ,
                        "trace_index": None,  # Will be set by the caller
                        "timestamp": convert_timestamp(time.time()),# convert to min/hour/day/month/year
                        "generate_config":{
                            "model": model,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "top_p": top_p,
                            "top_k": top_k

                        }
                    }
                }
            except Exception as e:
                retries += 1
                logger.warning(f"API request failed (attempt {retries}/{max_retries}): {e}")
                
                if retries < max_retries:
                    sleep_time = retry_delay * (2 ** (retries - 1)) * (0.5 + random.random())
                    logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    raise
        
        raise Exception("Maximum retries exceeded")

def get_lock_path(cache_dir, question_hash, trace_index):
    """Get the path to a lock file."""
    return os.path.join(cache_dir, f"{question_hash}_{trace_index}.lock")

def acquire_lock(cache_dir, question_hash, trace_index):
    """Try to acquire a lock file."""
    lock_path = get_lock_path(cache_dir, question_hash, trace_index)
    try:
        with open(lock_path, 'x') as f:
            f.write(f"{time.time()}")
        return True
    except FileExistsError:
        return False

def release_lock(cache_dir, question_hash, trace_index):
    """Release a lock file."""
    lock_path = get_lock_path(cache_dir, question_hash, trace_index)
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass

def get_response_path(output_dir, question_hash, trace_index):
    """Get the path to a response file."""
    return os.path.join(output_dir, f"{question_hash}_{trace_index}.json")

def is_response_generated(output_dir, question_hash, trace_index):
    """Check if a response has already been generated."""
    response_path = get_response_path(output_dir, question_hash, trace_index)
    if os.path.exists(response_path):
        try:
            with open(response_path, 'r') as f:
                data = json.load(f)
                # Check if the response contains required fields
                if "response" in data and "response_information" in data:
                    return True
        except (json.JSONDecodeError, IOError):
            # If the file is corrupt, we'll regenerate it
            pass
    return False
def convert_timestamp(timestamp):
    """Convert a timestamp to a human-readable format."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

def get_last_boxed(text: str):
    """Extract the last boxed expression (\\boxed{...}) from the text.

    Args:
        text (str): The text to check

    Returns:
        str: The last boxed expression, or None if not found
    """
    start_idx = text.rfind("\\boxed")
    if start_idx < 0:
        return None

    right_brace_idx = None
    num_left_braces_open = 0
    for i in range(start_idx, len(text)):
        if text[i] == "{":
            num_left_braces_open += 1
        if text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

    if not right_brace_idx:
        return None
    return text[start_idx : right_brace_idx + 1].replace("\\boxed", "").replace("\\text", "").replace("{", "").replace("}", "").strip()  


def extract_boxed(response):
    solution = response
    if solution is None:
        return None
    res= get_last_boxed(solution)
    if res is not None:
        return res
    try:
        return ok(response)
    except:
        return None

def ok(response):
    solution = str(response)
    if "The answer is" in solution:
        value = solution.split("The answer is")[1].split(".")[0].strip()
        if value.strip().upper() in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]:
            return value 
        if value[0] == '(' and len(value) >=3 and value[2] == ')' and value[1].upper() in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]:
            return value[1]
    return None

def check_action(row):
    # response = row['response'].split("Okay, let’s think step by step to solve the question by myself first. This will help me create a clear plan to solve the problem through searching and visiting pages afterward.")[-1].strip()
    assisst = []
    for r in row['list_action']:
        if r['role'] == 'assistant' and r['content_type'] != 'prefill-text' and r['content_type']!='code-response':
            assisst.append(r['content'])
    response = "\n".join(assisst)
    has_repeats_clean, repeated_items_clean = check_for_repeated_sections(response)
    return {
        "has_repeat": has_repeats_clean,
        "repeated_items_clean": repeated_items_clean
    }

def solve_problem(
    question,
    model,
    url_api,
    max_token,
    temperature
):
    tool = WebSearchTool()
    tool2 = VisitWebpageTool()
    descriptions = [
        tool.format_description(),
        tool2.format_description()
    ]
    available_tools = """
web_search: Performs a google web search based on your queries (think a Google search) then returns the top search results but only the title, url and a short snippet of the search results. To get the full content of the search results, you MUST use the visit_webpage.
Takes inputs: {'query': {'type': 'string', 'description': 'The query string'}, 'max_result': {'type': 'int', 'description': 'The maximum result to request'}}
Returns an output of type: str
visit_webpage: You should call this tool when you need to visit a webpage and extract its content. Returns webpage content as text.
Takes inputs: {'type': 'object', 'properties': {'url': {'type': 'string', 'description': 'Retrieves the content of a webpage by accessing the specified URL. This tool simulates a visit to the website and returns the full HTML source code of the page as a string'}}, 'required': ['url']}
Returns an output of type: str
    """
    list_action = []
    
    list_action.append(
        {
                "role": "system",
                "content": AgentConfig().system_prompt.format(current_date=datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT"),
                tool=available_tools),
                "content_type": "text",
                "code_block": "",
                "time-excute": -1,
                "has_think": False
        }
    )
    list_action.append(
        {
            "role": "user",
            "content": question,
            "content_type": "text",
            "code_block": "",
            "time-excute": -1,
            "has_think": False
        }
    )
    current_question =init_ques= apply_chat_template([
            {
                "role": "system",
                "content": AgentConfig().system_prompt.format(current_date=datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT"),
                tool=available_tools)
            },
            {
                "role": "user",
                "content": question
            }
        ])  +  AgentConfig().instructions
    # print(current_question)
    # exit(0)
    list_action.append(
        {
            "role": "assistant",
            "content": AgentConfig().instructions,
            "content_type": "prefill-text",
            "code_block": "",
            "time-excute": -1,
            "has_think": False
        }
    )
    total_token = 0
    is_complete = True
    stop = ["<end_code>"]
    total_call = 0
    target_max_token_states = [
        8192,
        12384,
        14000,
        max_token
    ]
    current_max_token_idx = len(target_max_token_states)-1
    for step in range(64):
        m = target_max_token_states[current_max_token_idx]
        if m-total_token <=0:
            is_complete=False
            break
        
        output = call_api(
            current_question,
            model, url_api,
            other_kwargs={"stop": stop},
            max_tokens=m-total_token,
            temperature=temperature
        )
        is_stop_by_length=output['response_information']['is_stop_by_length']
        if is_stop_by_length:
            if m == max_token:
                is_complete=False
                break
            else:
                # check repeat token 
                cnt_think=output['response'].count("</think>")
                has_repeats_clean, repeated_items_clean = check_for_repeated_sections(output['response'])
                if has_repeats_clean or cnt_think>2:
                    print("we found the repeat text!!! so it's not valid", repeated_items_clean)
                    current_question = current_question + output['response']
                    list_action.append(
                        {
                        "role": "assistant",
                        "content": output['response'],
                        "content_type": "text",
                        "code_block": "",
                        "time-excute": -1,
                        "has_think": "</think>" in output['response'],
                        }
                    )
                    is_complete=False
                    break
                else:
                    current_max_token_idx += 1 # we will increase the max-token and continue decode
                    current_question = current_question + output['response']
                    list_action.append(
                        {
                        "role": "assistant",
                        "content": output['response'],
                        "content_type": "text",
                        "code_block": "",
                        "time-excute": -1,
                        "has_think": "</think>" in output['response'],
                        }
                    )
                    continue

                
        total_token = output['response_information']['total_tokens']
        # response = output['response']
        matched_stop = output['response_information']['matched_stop']
        if matched_stop == "<end_code>":
            total_call+=1
            response = output['response']
            def strip_res(res):
                if len(res) > len("<end") and res[-1*len("<end"):] == "<end":
                    return res[:-1*len("<end")]
                return res
            
            if ConfigConstants.CODE_BLOCK_START in response:
                code_block  = response.split(ConfigConstants.CODE_BLOCK_START)[1].split(ConfigConstants.CODE_BLOCK_END)[0]
            else:
                code_block = response.split("<start_code>")[-1].split(ConfigConstants.CODE_BLOCK_END)[0]
            output['response'] = strip_res(output['response'])
            start_time = time.time()
            tool_name='web_search'
            if "web_search" in code_block:
                excute_output = tool.execute(code_block)
                if "Error:" in excute_output:
                    logger.info(f'tool-call ({tool_name}) failed within: {excute_output} log and code block is {code_block} ')
                    total_call-=1
                    continue
            else:
                tool_name='visit'
                excute_output = tool2.execute(code_block)
                # We need check visit_page 
                if "Error:" in excute_output:
                    # We need to fall back to previous steps by simple continue
                    logger.info(f'tool-call ({tool_name}) failed within: {excute_output} log and code block is {code_block} ')
                    total_call-=1
                    continue


            end_time = time.time()
            total_time =end_time-start_time
            logger.info(f'tool-call ({tool_name}) successfull within: {total_time} seconds ')
            
            if "web_search" in code_block:
                prefix = "This results may not enough to provide useful information, or just some partial information. I think step by step and check if it has any part useful, if so I will use visit_webpage to check more detail\n"
                prefix = "I'm remembering that the requirement of Part 2 with section 2.3 **Execute & Refine (Critical Loop)** about the **The Core Mandate** and **Refine**"
                prefix= "I understand that web search results only provide preview text and truthful url, so to access the full content, I need to use visit_webpage. I will check the information, and if any related pages or partial related pages are found, visit them to confirm and gather more information. "
                # prefix="This results may not enough to provide useful information.\nI must do more research or use visit_webpage function to get detailed information."
            else:
            
                prefix=("Okay, Let's review the content from page, find the useful information and think step by step to make the next plan.")
                T
            import numpy as np
            prefixs=[
                "\nLet's carefully analyze each part of the problem before proceeding to the next step.\n",

                # "\nLet's break this down into logical steps and address each one methodically.\n",

                # "\nLet’s reason through this gradually to build a complete solution.\n",

                "\nLet's think critically about what we know and what we need to figure out.\n",

                "\nLet’s use structured thinking to move forward with solving this question.\n",

                # "\nLet’s identify key variables and relationships before we proceed further.\n",

                "\nLet’s apply deductive reasoning to understand the next step clearly.\n",

                # "\nLet’s consider possible approaches and evaluate which one suits the problem best.\n",

                # "\nLet’s focus on one logical inference at a time to ensure clarity in solving.\n",

                # "\nLet’s walk through this systematically and validate each step as we go.\n",

                "\nLet’s reflect on what we’ve concluded so far and how it informs our next move.\n",

                "\nLet’s reason from the known information and extend it to uncover the unknown.\n",

                # "\nLet’s formulate hypotheses and test them step by step against the problem.\n",

                # "\nLet’s build a mental model of the problem and navigate through it thoughtfully.\n",

                "\nLet’s reason jointly using available data and logic to solve the problem effectively.\n",
                
            ]
            # prefix = "\nLet's think step by step to solve the question.\n"
            np.random.shuffle(prefixs)
            # prefix = prefixs[0]
            prefix=""
            current_question = current_question + output['response'].replace("</think>", "") + "<end_code>"  + "\n```output\n" + excute_output + "\n```\n" + prefix
            current_question = current_question.replace(
                "<end_code<end_code>", "<end_code>"
            )
            list_action.append(
                {
                "role": "assistant",
                    "content": output['response'].replace("</think>", ""),
                    "has_think": "</think>" in output['response'],
                    "time-excute": -1,
                    "code_block": "",
                    "content_type": "text"
                }
            )
            list_action.append(
                {
                    "role": "assistant",
                    "content": excute_output,
                    "content_type": "code-response",
                    "code_block": code_block,
                    "time-excute": total_time,
                    "has_think": False
                }
            )
            if prefix!="":
                list_action.append(
                    {
                        "role": "assistant",
                        "content": prefix,
                        "content_type": "prefill-text",
                        "code_block": "",
                        "time-excute": 0,
                        "has_think": False
                        
                    }
                )
        elif matched_stop =="</think>":
            # if total_call == 0:
            #     current_question = init_ques #+  "\n\nI'm remembering that I must follow the instruction and I can access the python code to support my reasoning."
            #     # list_action.append(
            #     #     {
            #     #         "role": "assistant",
            #     #         "content": output['response'].replace("</think>",""),
            #     #         "content_type": "prefill-text",
            #     #         "code_block": "",
            #     #         "time-excute": 0,
            #     #         "has_think": False
                        
            #     #     }
            #     # )
            #     list_action.append(
            #         {
            #             "role": "assistant",
            #             "content": "\n\nI'm remembering that I must follow the instruction and I can access the python code to support my reasoning",
            #             "content_type": "prefill-text",
            #             "code_block": "",
            #             "time-excute": 0,
            #             "has_think": False
                        
            #         }
            #     )
            # else:
            current_question = current_question + output['response'] + "</think>"
            list_action.append(
                    {
                        "role": "assistant",
                        "content": output['response'],
                        "content_type": "text",
                        "code_block": "",
                        "time-excute": 0,
                        "has_think": False
                        
                    }
                )
            stop = ['<end_code>']

        else:
            # print(matched_stop)
            
            current_question = current_question + output['response']
            list_action.append(
                {
                "role": "assistant",
                "content": output['response'],
                "content_type": "text",
                "code_block": "",
                "time-excute": -1,
                "has_think": "</think>" in output['response'],
                }
            )
            break

    # response = row['response']
    last_think =current_question.split("</think>")[-1].strip()
    extract_boxed_think = extract_boxed(last_think)
    return {
        "list_action": list_action,
        "response": current_question,
        "is_complete": is_complete,
        "extracted_answer": extract_boxed_think,
        "model": model,
        "generation_config": {
            "max-token": max_token,
            "temperature": temperature
        },
        "total_tokens": total_token
    }


def process_question(args):
    """Process a single question and trace."""
    question_file, question_hash, trace_index, output_dir, cache_dir, base_url, model, config = args
    
    # Check if this response is already generated
    if is_response_generated(output_dir, question_hash, trace_index):
        logger.info(f"Response already exists for {question_hash}, trace {trace_index}")
        return True
    
    # Try to acquire a lock
    if not acquire_lock(cache_dir, question_hash, trace_index):
        logger.info(f"Another process is handling {question_hash}, trace {trace_index}")
        return False
    
    try:
        # Load the question
        if isinstance(question_file, str):
            with open(question_file, 'r') as f:
                question_data = json.load(f)
        else:
            question_data = question_file
        
        question_text = question_data.get("question", "")
        
        # Call the API
        logger.info(f"Generating response for {question_hash}, trace {trace_index}")
        list_failed = []

        def check_success(response):
            # return True, ""
            has_code_content = False
            has_empty_code = False
            has_empty_response_code = False
            has_error_code = False
            error_code = ""
            list_action = response['list_action']
            for act in list_action:
                if act.get("content_type", "text") == "code-response":
                    has_code_content = True
                    if str(act['code_block']).strip() == "":
                        has_empty_code = True
                    if str(act['content']).strip() == "" or "Code executed successfully (no output)" == str(act['content']).strip():
                        has_empty_response_code = True
                    if "Error:" in str(act['content']).strip():
                        has_error_code = True
                        error_code = str(act['content']).strip() + "\n\n" + act['code_block']
            is_last_empty = list_action[-1]['content'].strip() == ""
            if has_code_content == False:
                return False, "model didn't call code???"
            if has_empty_code == True:
                return False, "Model return empty code with unknow why???"
            if has_empty_response_code == True:
                return False, "The code which was generated by model has empty output with unknow why"
            if has_error_code == True:
                return False, f"The code which was generated by model has error code: {error_code}"
            if is_last_empty == True:
                return False, f'The last response LLM is empty, maybe reason is due to has another </think> before'
            return True, "Success"
        valid = False
        for it in range(4):
            
            response_data = solve_problem(question_text, model, base_url, max_token=config.max_token, temperature=config.temperature)
            status, log_check = check_success(response_data)
            if status:
                valid = True
                break
            else:
                list_failed.append(
                    {
                        k:v for k,v in response_data.items()
                    }
                )
                logger.error(
                    f"Error check response data {question_hash}, trace {trace_index}: {log_check} at {it}/4 retry",
                    exc_info=True
                )
        response_data['valid'] = valid
        response_data['list_failed'] = list_failed
        response_data['response_information'] = {}
        response_data["response_information"]["trace_index"] = trace_index
        
        # Combine question and response data
        output_data = {**question_data, **response_data}
        
        # Save the response
        response_path = get_response_path(output_dir, question_hash, trace_index)
        with open(response_path, 'w') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Successfully processed {question_hash}, trace {trace_index}")
        return True
    
    except Exception as e:
        logger.error(f"Error processing {question_hash}, trace {trace_index}: {e}", exc_info=True)
        return False
    
    finally:
        # Always release the lock
        release_lock(cache_dir, question_hash, trace_index)

def init_worker(semaphore):
    """Initialize worker process with the semaphore."""
    global api_semaphore
    api_semaphore = semaphore

def get_model(base_url):
    client = openai.Client(base_url=base_url, api_key="EMPTY")
    
    model_name = client.models.list()#.data[0].id
    if len(model_name.data) == 0:
        raise ValueError("No model found")
    model_name = model_name.data[0].id
    return model_name

def main():
    parser = argparse.ArgumentParser(description="Generate LLM responses in parallel")
    # parser.add_argument("--dataset-name", type=str, required=True, 
    #                     help="The dataset name or path of HF dataset, which contains the question column and hash_value column and need_to_gen column")
    parser.add_argument("--output-dir", type=str, required=True, 
                        help="Directory to store response files")
    parser.add_argument("--cache-dir", type=str, required=True, 
                        help="Directory for lock files")
    parser.add_argument("--base-url", type=str, default="http://localhost:30000/v1",
                        help="Base URL for the OpenAI API")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        help="Model to use for generation")
    parser.add_argument("--num-workers", type=int, default=64, 
                        help="Number of worker processes")
    parser.add_argument("--num-traces", type=int, default=1, 
                        help="Number of traces to generate per question")
    parser.add_argument("--max-api-calls", type=int, default=1024,
                        help="Maximum number of concurrent API calls")
    parser.add_argument("--node-id", type=int, default=0,
                        help="Node ID")
    parser.add_argument("--num-nodes", type=int, default=1,
                        help="Number of nodes")
    parser.add_argument("--shuffle-before-generate", type=bool, default=True,
                        help="Shuffle the work items before generating")
    parser.add_argument("--force-overwire-num-traces", type=int, default=0,
                        help="Force overwire the number of traces to generate")
    
    parser.add_argument("--max-token", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)

    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    # os.system(f"rm -rf {args.cache_dir}")
    os.makedirs(args.cache_dir, exist_ok=True)
    while True:
        try:
            args.model = get_model(args.base_url)
            break
        except Exception as e:
            logger.error(f"Error getting model: {e}")
            import time
            time.sleep(30)
    # model
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    from datasets import load_dataset, load_from_disk
    import numpy as np
    dataset_generate = load_dataset("meoconxinhxan/search_r1_ds")['train']#.select(range(4))
    # dataset_generate=dataset_generate.select(range(20000, 40000))
    work_items = []
    from tqdm import tqdm
    import glob
    list_accepted = glob.glob(args.output_dir + "/*.json")

    # list_json = [json.load(open(i)) for i in list_accepted]
    list_json = []
    for i in tqdm(list_accepted):
        try:
            list_json.append(json.load(open(i)))
        except:
            list_json.append("failed")
    index_valid = []
    for row, path in zip(list_json, list_accepted):
        if isinstance(row, str) and row=='failed':
            # os.system(f"rm {path}")
            continue
        index_in_ds = row['other_information']['index_in_ds']
        response = row['response']
        if "Server Error" not in response:
            index_valid.append(index_in_ds)
        else:
            os.system(f"rm {path}")
    


    print(len(index_valid), len(list_json))
    for index, row in tqdm(enumerate(dataset_generate), total=len(dataset_generate)):
        if index in index_valid:
            continue
        question = row['prompt'][-1]['content']
        question = question.strip()
        if question[-1] != '?':
            question += '?'
        row['question'] = question
        if "boxed" not in row['question']:
            row['question'] = row['question' ] + "\n\nPlease reason step-by-step, and put your final answer within \\boxed{}."
        question_file = {
            "question": row['question'],
            "other_information": {
                "index_in_ds": index,
                "golden_answers": row['reward_model']['ground_truth'],
            }
        }
        if 'hash_value' not in row:
            hash_value = hashlib.sha256(row['question'].encode('utf-8')).hexdigest()
            row['hash_value'] = hash_value
        question_hash = row['hash_value']
        if 'need_to_gen' not in row:
            row['need_to_gen'] = args.num_traces
        need_to_gen = row['need_to_gen']
        if args.force_overwire_num_traces > 0:
            need_to_gen = args.num_traces
        for trace_index in range(need_to_gen):
            work_items.append((
                    question_file, 
                    question_hash, 
                    trace_index, 
                    args.output_dir, 
                    args.cache_dir, 
                    args.base_url, 
                    args.model,
                    args
                ))
    print(work_items[0])
    work_items = sorted(work_items, key=lambda x: x[1])
    work_items=work_items[::-1] # reverse
    np.random.shuffle(work_items)
    process_question(work_items[0])
    # exit(0)
    import numpy as np
    work_items = np.array_split(work_items, args.num_nodes)[args.node_id]
    logger.info(f"Total work items: {len(work_items)}")
    max_concurrent_calls = min(args.max_api_calls, args.num_workers)
    semaphore = multiprocessing.Semaphore(max_concurrent_calls)
    logger.info(f"Limiting to {max_concurrent_calls} concurrent API calls")
    import time
    time.sleep(10) # sleep for 32 seconds to wait another nodes
    # Process questions in parallel
    if len(work_items):
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=init_worker,
            initargs=(semaphore,)
        ) as executor:
        
            results = list(executor.map(process_question, work_items))
    
    # Report results
    successful = results.count(True)
    logger.info(f"Completed {successful} out of {len(work_items)} work items")


if __name__ == "__main__":
    main()

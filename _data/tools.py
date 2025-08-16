import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict
from code_exe import PythonExecutor
import re
import sys
import subprocess
import tempfile
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import traceback
import time
import numpy as np

SEARCH_SUFFIX = ("This results may not enough to provide useful information. "
                     "I must do more research or use page_visit tool to get detailed information. \n")
class BaseTool(ABC):
    """Base class for all tools."""

    name: str
    description: str
    argument_schema: Dict[str, Dict[str, Any]]
    return_type: str
    suffix: str

    @abstractmethod
    def execute(self, tool_history = None, **kwargs) -> str:
        """Execute the tool with the given arguments."""

    @classmethod
    @abstractmethod
    def reset(cls) -> None:
        """Reset Tool State"""
        pass

    def execute_stream(self,
                             stream_event: Callable[[str, Dict[str, Any]], None],
                             tool_history = None,
                             **kwargs) -> str:
        """Execute the tool with the given arguments."""
        stream_event("tool", {"name": self.name, "arguments": kwargs})
        # asyncio.sleep(0)
        result = self.execute(tool_history, **kwargs)
        return result

    def format_description(self) -> str:
        """Format the tool description for the LLM."""
        return f"- {self.name}: {self.description}\n    Takes inputs: {self.argument_schema}\n    Returns an output of type: {self.return_type}"


class WebSearchTool(BaseTool):
    """Tool for performing web searches."""

    name = "web_search"
    description = "Performs a google web search based on your queries (think a Google search) then returns the top search results but only the title, url and a short snippet of the search results. To get the full content of the search results, you MUST use the visit_webpage."
    argument_schema = {
        "query": {
            "type": "string",
            "description": "The query string",
        },
        "max_result": {
            "type": "int",
            "description": "The maximum result to request"
        }
    }
    return_type = "str"
    suffix = SEARCH_SUFFIX

    # Set to store already searched queries
    _searched_queries = set()

    @classmethod
    def reset(cls) -> None:
        """Reset the set of searched queries."""
        cls._searched_queries = set()

    def execute(self, response) -> str:
        """Execute the web search."""
        prefix="""import os,sys,json\nsys.path.append("/home/slurm/hoanganh/search_agent/_benchmark")\nfrom tool_interface import *"""
        code = prefix + "\n" + response
        print(code)
        executor = PythonExecutor(get_answer_from_stdout=True, timeout_length=512)
        predictions =execute_code(code)
        # print(code)
        output_execute = str(predictions[0])
        if output_execute=="":
            print(predictions[1])
        return output_execute



def execute_code(code: str):
    """Execute Python code safely and return output."""
    try:
        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        # Execute the code with timeout
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=256
        )
        
        # Clean up
        os.unlink(temp_file)
        
        if result.returncode == 0:
            output = result.stdout.strip()
            return output if output else "Code executed successfully (no output)", True
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return f"Error: {error_msg}", False
            
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (256 seconds)", False
    except Exception as e:
        return f"Error: {str(e)}", False


import io
import sys
from contextlib import redirect_stdout

def execute_trusted_code(code: str):
    """
    Execute trusted Python code and return the output.
    WARNING: This is NOT safe for untrusted code!
    """
    if "os.system(" in code:
        return f"Error: Code has os.system('') calling, very dangerous!!!", False
    output_buffer = io.StringIO()
    try:
        with redirect_stdout(output_buffer):
            exec(code, {}) # Using an empty dict for the global scope
        output = output_buffer.getvalue().strip()
        return output if output else "Code executed successfully (no output)", True
    except Exception as e:
        return f"Error: {str(e)}", False

class VisitWebpageTool(BaseTool):
    name = "visit_webpage"
    description = "You should call this tool when you need to visit a webpage and extract its content. Returns webpage content as text."
    argument_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Retrieves the content of a webpage by accessing the specified URL. "
                   "This tool simulates a visit to the website and returns the full HTML "
                   "source code of the page as a string",
            }
        },
        "required": ["url"],
    }
    output_type = "string"
    suffix = SEARCH_SUFFIX
    return_type = "str"
    def __init__(self):
        pass 

    def execute(self, response) -> str:
        prefix="""import os,sys,json\nsys.path.append("/home/slurm/hoanganh/search_agent/_benchmark")\nfrom tool_interface import *"""
        code = prefix + "\n" + response
        print(code)
        executor = PythonExecutor(get_answer_from_stdout=True, timeout_length=40)
        predictions = execute_code(code)
        output_execute = str(predictions[0])
        if output_execute == "":
            print("Error call code")
            print(code)
            print(predictions[1])
            print('--------')
        return output_execute

    @classmethod
    def reset(cls) -> None:
        """Reset the set of searched queries."""
        cls._searched_queries = set()

    
if __name__ == "__main__":
    # code_test="print(web_search('genetic causes of arrhythmogenic right ventricular dysplasia (ARVD)'))"
    code_test2="print(visit_webpage('https://www.hopkinsmedicine.org/health/conditions-and-diseases/arrhythmogenic-right-ventricular-dysplasia--cardiomyopathy-arvdc'))"
    # print(WebSearchTool().execute(
    #     code_test
    # ))
    print(VisitWebpageTool().execute(code_test2))

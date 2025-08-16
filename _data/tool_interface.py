import os
import requests
import numpy as np
import time
import json
import http


# Local search endpoint
def call_search(query_data, max_result=10):
    current_time_seconds = int(time.time()) 
    xx=len(query_data)
    xx=int(xx) % 5
    search_endpoint = "http://0.0.0.0:10000/search"
    current_time_seconds=current_time_seconds% 5
    current_time_seconds=int(current_time_seconds)
    response = requests.post(search_endpoint, json={"query": query_data, "use_reranker": False, "preview_char": 512, "top_k": max_result})
    values =  response.json()['results'][:max_result]
    str_format = []
    for index, res in enumerate(values):
        paper_title = res['metadata']['paper_title']
        url_paper = res['url']
        preview = res['preview']
        str_format.append(f'''\nIndex: {index}\nURL: {url_paper}\nTitle: {paper_title}\nPreview: {preview}\n''') #\\nFirst 100 words: {first_200_words}

    return ''.join(str_format)

# def format_web_search_response(data) -> str:
#     """Format web_search response to show indexed results with URL and Title."""
#     try:
#         if isinstance(data, dict) and 'results' in data and data['results']:
#             formatted_lines = []
#             for idx, result in enumerate(data['results']):
#                 url = result.get('url', 'N/A')
#                 title = result.get('metadata', {}).get('paper_title', 'No title')
#                 preview = result.get('preview', 'No preview available')
#                 formatted_lines.append(f"Index: {idx}")
#                 formatted_lines.append(f"URL: {url}")
#                 formatted_lines.append(f"Title: {title}")
#                 formatted_lines.append(f"Preview: {preview}")
#                 if idx < len(data['results']) - 1:
#                     formatted_lines.append("")  # Empty line between results
#             return "\n".join(formatted_lines)
#         else:
#             return str(data)
#     except Exception as e:
#         return f"Error formatting web_search response: {str(e)}"

# # Serpdev
# def call_search(query_data, max_result=10):
#     """Perform a web search using Serper API."""
#     conn = None
#     try:
#         conn = http.client.HTTPSConnection("google.serper.dev", timeout=30)
#         payload = json.dumps({
#             "q": query_data,
#             "num": 10
#         })
#         headers = {
#             'Content-Type': 'application/json'
#         }

#         conn.request("POST", "/search", payload, headers)
#         res = conn.getresponse()
#         raw_data = res.read()
        
#         # Check if response is valid
#         if res.status != 200:
#             return {'results': [], 'error': f"HTTP Error {res.status}: {res.reason}"}
        
#         result = json.loads(raw_data.decode("utf-8"))

#         # Format the results
#         formatted_results = {
#             'results': []
#         }
#         if "organic" in result and result["organic"]:
#             for item in result["organic"]:
#                 snippet = item.get('snippet', 'No preview available')
                    
#                 formatted_results['results'].append({
#                     'url': item.get("link", ""),
#                     'metadata': {
#                         'paper_title': item.get('title', 'No title')
#                     },
#                     'preview': snippet
#                 })    
        
#         return formatted_results
#     except json.JSONDecodeError as e:
#         return {'results': [], 'error': f"JSON parsing error: {str(e)}"}
#     except Exception as e:
#         return {'results': [], 'error': f"Error performing web search: {str(e)}"}
#     finally:
#         if conn:
#             conn.close()


def web_search(query, max_result=10):
    return call_search(query,max_result)
    # search_response = call_search(query, max_result)
    # return format_web_search_response(search_response)


# Local visit endpoint
def call_visit(url):
    visit_endpoint = "http://0.0.0.0:10000/visit"
    response = requests.post(visit_endpoint, json={"url" : url})
    return response.json()['data']


# def clean_web_content(content: str) -> str:
#     """Clean web content by removing URLs, extra whitespace, and limiting length."""
#     import re
    
#     if not content:
#         return ""
    
#     # Remove URLs (http/https/ftp/www links)
#     url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]*|www\.[^\s<>"{}|\\^`\[\]]*|ftp://[^\s<>"{}|\\^`\[\]]*'
#     content = re.sub(url_pattern, '', content)
    
#     # Remove email addresses
#     email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
#     content = re.sub(email_pattern, '', content)
    
#     # Remove markdown links [text](url)
#     markdown_link_pattern = r'\[([^\]]*)\]\([^)]*\)'
#     content = re.sub(markdown_link_pattern, r'\1', content)
    
#     # Remove excessive whitespace, newlines, and special characters
#     content = re.sub(r'\s+', ' ', content)  # Replace multiple whitespace with single space
#     content = re.sub(r'\n+', '\n', content)  # Replace multiple newlines with single newline
#     content = re.sub(r'[\t\r\f\v]+', ' ', content)  # Replace tabs and other whitespace
    
#     # Remove common navigation/footer text patterns
#     noise_patterns = [
#         r'Skip to main content',
#         r'Privacy Policy',
#         r'Terms of Service',
#         r'Cookie Policy',
#         r'© \d{4}',
#         r'All rights reserved',
#         r'Subscribe to newsletter',
#         r'Follow us on',
#         r'Share this',
#         r'Print this page'
#     ]
    
#     for pattern in noise_patterns:
#         content = re.sub(pattern, '', content, flags=re.IGNORECASE)
    
#     # Strip leading/trailing whitespace
#     content = content.strip()
    
#     # Cap at 10000 characters
#     if len(content) > 10000:
#         content = content[:10000] + "... [Content truncated]"
    
#     return content


# # Serpdev
# def call_visit(url: str) -> dict:
#     """Visit a webpage and extract its content using Serper API."""
#     conn = None
#     try:
#         conn = http.client.HTTPSConnection("scrape.serper.dev", timeout=30)
#         payload = json.dumps({
#             "url": url,
#             "includeMarkdown": True
#         })
#         headers = {
#             'Content-Type': 'application/json'
#         }

#         conn.request("POST", "/", payload, headers)
#         res = conn.getresponse()
#         raw_data = res.read()
        
#         # Check if response is valid
#         if res.status != 200:
#             return {'data': f"HTTP Error {res.status}: {res.reason}"}
        
#         result = json.loads(raw_data.decode("utf-8"))

#         # Extract text content
#         raw_content = ""
#         if "markdown" in result and result["markdown"]:
#             raw_content = result["markdown"]
#         elif "text" in result and result["text"]:
#             raw_content = result["text"]
#         else:
#             raw_content = json.dumps(result, indent=2)

#         # Clean and process the content
#         cleaned_content = clean_web_content(raw_content)
        
#         return {'data': cleaned_content}
#     except json.JSONDecodeError as e:
#         return {'data': f"JSON parsing error: {str(e)}"}
#     except Exception as e:
#         return {'data': f"Error visiting webpage: {str(e)}"}
#     finally:
#         if conn:
#             conn.close()


def visit_webpage(url):
    return call_visit(url)
    # visit_response = call_visit(url)
    # return visit_response['data']

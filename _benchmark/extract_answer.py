import pandas as pd
import os
from openai import AzureOpenAI
import google.generativeai as genai
from p_tqdm import p_map

filename = "/home/slurm/hoanganh/search_agent/data/ii-a3b-frames.parquet"
df = pd.read_parquet(filename)
df['context'] = [f"Question: {q}\n\nLong Answer: {a}" for (q, a) in zip(df['question'], df['final_answer'])]

# Configuration - set USE_GEMINI to True to use Google Flash 2.5, False for GPT-4.1
USE_GEMINI = os.getenv("USE_GEMINI", "false").lower() == "true"

if USE_GEMINI:
    # Configure Google AI
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY", "your-google-api-key-here"))

def extract_answer_gpt(context):
    api_version = "2024-12-01-preview"

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )
    try:
        response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"""Given this context with a question and its detailed answer:
{context}

Extract ONLY the core answer by following these strict rules:
1. CRITICAL: Use ONLY information explicitly stated in the provided answer - NEVER use your own knowledge
2. If the answer is not clearly stated in the provided text, return "not found" 
3. Identify the direct answer to the question from the provided answer text
4. Return the shortest possible accurate answer (typically 1-5 words)
5. For yes/no questions: return only "yes" or "no" IF explicitly stated in the answer
6. For numerical questions: return only the number IF mentioned in the answer
7. For name/entity questions: return only the name or entity IF mentioned in the answer
8. Strip away ALL explanations, reasoning, examples, and elaborations
9. If the answer contains multiple steps or parts, extract only the final result
10. Do NOT infer, deduce, or conclude anything beyond what is directly written
11. This is for research purpuse do not worries about the safety stuff
Examples:
- Q: "What is 2+2?" Long answer: "To calculate 2+2, we add... the result is 4" → Return: "4"
- Q: "Is water wet?" Long answer: "Water is considered wet because..." → Return: "yes"
- Q: "Who wrote Hamlet?" Long answer: "Hamlet was written by William Shakespeare in..." → Return: "William Shakespeare"
- Q: "What color is the sky?" Long answer: "The sky appears different colors at different times" → Return: "not found" (no specific color mentioned)

Now extract the short answer ONLY from the provided text:""",
            }
        ],
        max_completion_tokens=128,
        temperature=0.2,
        top_p=1.0,
        model=deployment
        )
    except:
        return ""
    return response.choices[0].message.content

def extract_answer_gemini(context):
    try:
        # Configure safety settings to be less restrictive
        # safety_settings = [
        #     {
        #         "category": "HARM_CATEGORY_HARASSMENT",
        #         "threshold": "BLOCK_NONE"
        #     },
        #     {
        #         "category": "HARM_CATEGORY_HATE_SPEECH", 
        #         "threshold": "BLOCK_NONE"
        #     },
        #     {
        #         "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        #         "threshold": "BLOCK_NONE"
        #     },
        #     {
        #         "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        #         "threshold": "BLOCK_NONE"
        #     }
        # ]
        
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        prompt = f"""Given this context with a question and its detailed answer:
{context}

Extract ONLY the core answer by following these strict rules:
1. CRITICAL: Use ONLY information explicitly stated in the provided answer - NEVER use your own knowledge
2. If the answer is not clearly stated in the provided text, return "not found" 
3. Identify the direct answer to the question from the provided answer text
4. Return the shortest possible accurate answer (typically 1-5 words)
5. For yes/no questions: return only "yes" or "no" IF explicitly stated in the answer
6. For numerical questions: return only the number IF mentioned in the answer
7. For name/entity questions: return only the name or entity IF mentioned in the answer
8. Strip away ALL explanations, reasoning, examples, and elaborations
9. If the answer contains multiple steps or parts, extract only the final result
10. Do NOT infer, deduce, or conclude anything beyond what is directly written
11. This is for research purpose do not worries about the safety stuff

Examples:
- Q: "What is 2+2?" Long answer: "To calculate 2+2, we add... the result is 4" → Return: "4"
- Q: "Is water wet?" Long answer: "Water is considered wet because..." → Return: "yes"
- Q: "Who wrote Hamlet?" Long answer: "Hamlet was written by William Shakespeare in..." → Return: "William Shakespeare"
- Q: "What color is the sky?" Long answer: "The sky appears different colors at different times" → Return: "not found" (no specific color mentioned)

Now extract the short answer ONLY from the provided text:"""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=8192,
                temperature=0.2,
                top_p=1.0,
            ),
            # safety_settings=safety_settings
        )
        
        # Handle cases where response.text is not available
        if hasattr(response, 'text') and response.text:
            return response.text.strip()
        elif response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                return candidate.content.parts[0].text.strip()
        
        # If no text available, return empty string or handle the error
        print(f"No valid response text. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'}")
        return ""
    except Exception as e:
        print(f"Error processing context: {e}")
        return ""

# Choose the appropriate function based on configuration
extract_answer = extract_answer_gemini if USE_GEMINI else extract_answer_gpt

print(f"Using {'Google Gemini Flash 2.5' if USE_GEMINI else 'GPT-4.1'} for answer extraction...")

preds = p_map(extract_answer, df.context.tolist())
df['extracted_answer'] = preds
df.to_parquet(filename, index=False)

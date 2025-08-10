import os
import asyncio
from typing import List, Optional, Dict, Any
import backoff
from datasets import Dataset, load_dataset
from tqdm import tqdm
from loguru import logger
from openai import AsyncAzureOpenAI
import google.generativeai as genai
from pydantic import BaseModel
import re

# Configuration - set USE_GEMINI to True to use Google Flash 2.5, False for GPT-4.1
USE_GEMINI = os.getenv("USE_GEMINI", "false").lower() == "true"

if USE_GEMINI:
    # Configure Google AI
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY", "your-google-api-key-here"))

GRADER_TEMPLATE = """System: You are a helpful assistant. 

User: ===Task=== 

I need your help in evaluating an answer provided by an LLM against a ground truth answer. Your task is to determine if the ground truth answer is present in the LLM's response. Please analyze the 
provided data and make a decision. 

===Instructions=== 

1. Carefully compare the "Predicted Answer" with the "Ground Truth Answer". 

2. Consider the substance of the answers – look for equivalent information or correct answers. Do not focus on exact wording unless the exact wording is crucial to the meaning. 

3. Your final decision should be based on whether the meaning and the vital facts of the "Ground Truth Answer" are present in the "Predicted Answer:" 

===Input Data=== 

- Question: {question} 

- Predicted Answer: {predicted_answer} 

- Ground Truth Answer: {target} 

===Output Format=== 

Provide your final evaluation in the following format: 

"Explanation:" (How you made the decision?) 

"Decision:" ("TRUE" or "FALSE" ) 

Please proceed with the evaluation.""".strip()


class Response(BaseModel):
    explanation: str
    decision: str


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=3,
    max_time=30,
    jitter=backoff.full_jitter
)
async def call_gpt_api_with_retry(client: AsyncAzureOpenAI, question: str, predicted_answer: str, target: str):
    """Call the GPT API with retry logic when failures occur."""
    return await client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "user", "content": GRADER_TEMPLATE.format(
                question=question,
                predicted_answer=predicted_answer,
                target=target)
            }
        ],
        temperature=0.0,
    )


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=3,
    max_time=30,
    jitter=backoff.full_jitter
)
async def call_gemini_api_with_retry(question: str, predicted_answer: str, target: str):
    """Call the Gemini API with retry logic when failures occur."""
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = GRADER_TEMPLATE.format(
        question=question,
        predicted_answer=predicted_answer,
        target=target
    )
    
    response = await asyncio.to_thread(
        model.generate_content,
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=8192,
            temperature=0.2,
            top_p=1.0,
        ),
    )
    return response


def extract_decision(response_text: str) -> str:
    """Extract TRUE/FALSE decision from the response text."""
    # Look for "Decision:" followed by TRUE or FALSE
    decision_match = re.search(r'Decision:\s*["\']?(TRUE|FALSE)["\']?', response_text, re.IGNORECASE)
    if decision_match:
        return decision_match.group(1).upper()
    
    # Fallback: look for TRUE or FALSE anywhere in the response
    if 'TRUE' in response_text.upper():
        return 'TRUE'
    elif 'FALSE' in response_text.upper():
        return 'FALSE'
    
    # Default to FALSE if unclear
    return 'FALSE'


async def process_item_gpt(idx: int, ds: Dataset, client: AsyncAzureOpenAI, 
                          question_column: str, response_column: str, 
                          ground_truth_column: str, out_column: str, pbar: tqdm) -> Dict[str, Any]:
    """Process a single item with GPT API and retries."""
    item = ds[idx]
    out_dict = dict(item)
    
    try:
        response = await call_gpt_api_with_retry(
            client=client,
            question=item[question_column],
            predicted_answer=item[response_column],
            target=item[ground_truth_column]
        )
        raw_response = response.choices[0].message.content
        out_dict[out_column] = raw_response
        out_dict[f"{out_column}_extracted"] = extract_decision(raw_response)
    except Exception as e:
        logger.error(f"Error processing item {idx} with GPT after retries: {str(e)}")
        out_dict[out_column] = f"Error: {str(e)}"
        out_dict[f"{out_column}_extracted"] = "FALSE"
    
    pbar.update(1)
    return out_dict


async def process_item_gemini(idx: int, ds: Dataset, 
                             question_column: str, response_column: str, 
                             ground_truth_column: str, out_column: str, pbar: tqdm) -> Dict[str, Any]:
    """Process a single item with Gemini API and retries."""
    item = ds[idx]
    out_dict = dict(item)
    
    try:
        response = await call_gemini_api_with_retry(
            question=item[question_column],
            predicted_answer=item[response_column],
            target=item[ground_truth_column]
        )
        
        raw_response = ""
        # Handle cases where response.text is not available
        if hasattr(response, 'text') and response.text:
            raw_response = response.text.strip()
        elif response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                raw_response = candidate.content.parts[0].text.strip()
            else:
                logger.warning(f"No valid response text for item {idx}. Finish reason: {candidate.finish_reason}")
                raw_response = "Decision: FALSE"  # Default for safety-filtered responses
        else:
            logger.warning(f"No candidates in response for item {idx}")
            raw_response = "Decision: FALSE"
        
        out_dict[out_column] = raw_response
        out_dict[f"{out_column}_extracted"] = extract_decision(raw_response)
    except Exception as e:
        logger.error(f"Error processing item {idx} with Gemini after retries: {str(e)}")
        out_dict[out_column] = f"Error: {str(e)}"
        out_dict[f"{out_column}_extracted"] = "FALSE"
    
    pbar.update(1)
    return out_dict


async def run(
    ds: Dataset, 
    question_column: str, 
    response_column: str, 
    ground_truth_column: str,
    out_column: str,
    client: Optional[AsyncAzureOpenAI] = None,
    max_threads: int = 10
) -> Dataset:
    """Process dataset with improved concurrency handling and retries."""
    logger.info(f"Processing dataset of size {len(ds)} using {'Gemini Flash 2.5' if USE_GEMINI else 'GPT-4.1'}")
    
    # Initialize progress bar
    pbar = tqdm(total=len(ds), desc="Evaluating items")
    
    # Process items in batches
    results = []
    for i in range(0, len(ds), max_threads):
        batch_indices = list(range(i, min(i + max_threads, len(ds))))
        
        if USE_GEMINI:
            tasks = [
                process_item_gemini(
                    idx, ds, 
                    question_column, response_column, 
                    ground_truth_column, out_column, pbar
                ) 
                for idx in batch_indices
            ]
        else:
            tasks = [
                process_item_gpt(
                    idx, ds, client, 
                    question_column, response_column, 
                    ground_truth_column, out_column, pbar
                ) 
                for idx in batch_indices
            ]
        
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
    
    pbar.close()
    return Dataset.from_list(results)


def calculate_accuracy(df, extracted_column: str = "is_correct_extracted"):
    """Calculate accuracy from extracted TRUE/FALSE decisions."""
    if extracted_column not in df.columns:
        logger.error(f"Column {extracted_column} not found in dataframe")
        return 0.0
    
    correct_count = sum(df[extracted_column] == "TRUE")
    total_count = len(df)
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    logger.info(f"Correct answers: {correct_count}")
    logger.info(f"Total answers: {total_count}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    return accuracy


async def main():
    from dotenv import load_dotenv
    load_dotenv()
    
    # Configure logging
    logger.add("evaluation_frames.log", rotation="10 MB")
    
    client = None
    if not USE_GEMINI:
        # Initialize the GPT client
        client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1"),
        )
    
    # Load data
    import pandas as pd
    filename = "/home/slurm/hoanganh/search_agent/data/ii-a3b-frames.parquet"
    print(filename)
    df = pd.read_parquet(filename)
    ds = Dataset.from_pandas(df)
    logger.info(f"Loaded dataset from parquet file: {len(ds)} items")
    
    # Run evaluation
    evaluated_ds = await run(
        ds=ds, 
        question_column="question", 
        response_column="extracted_answer", 
        ground_truth_column="ground_truth", 
        out_column="is_correct", 
        client=client, 
        max_threads=10
    )
    
    # Convert back to dataframe and calculate accuracy
    df_evaluated = evaluated_ds.to_pandas()
    
    # Calculate and display accuracy
    accuracy = calculate_accuracy(df_evaluated, "is_correct_extracted")
    
    # Update original dataframe with results
    df['is_correct'] = df_evaluated['is_correct']
    df['is_correct_extracted'] = df_evaluated['is_correct_extracted']
    
    # Save results
    df.to_parquet(filename)
    logger.info(f"Saved results to {filename}")
    
    print(f"Final Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    asyncio.run(main())

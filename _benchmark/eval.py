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

# Configuration - set USE_GEMINI to True to use Google Flash 2.5, False for GPT-4.1
USE_GEMINI = os.getenv("USE_GEMINI", "false").lower() == "true"

if USE_GEMINI:
    # Configure Google AI
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY", "your-google-api-key-here"))

GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". 
    - Predicted answers "120k", "124k", and 115k" are all CORRECT. 
    - Predicted answers "100k" and "113k" are INCORRECT. 
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
    - The presence or absence of commas in numbers (e.g., "5,876" vs "5876") does not affect grading.
    - Numbers written as words or digits are equivalent (e.g., "2 million" vs "2000000" vs "2,000,000" are all considered the same).
    - For large numerical answers, a margin of error of ±1% is acceptable (e.g., if the gold answer is 855, predicted answers between 846.45 and 863.55 are CORRECT).
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name. 
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()


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
        # safety_settings=safety_settings
    )
    return response


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
        out_dict[out_column] = response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error processing item {idx} with GPT after retries: {str(e)}")
        out_dict[out_column] = f"Error: {str(e)}"
    
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
        
        # Handle cases where response.text is not available
        if hasattr(response, 'text') and response.text:
            out_dict[out_column] = response.text.strip()
        elif response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                out_dict[out_column] = candidate.content.parts[0].text.strip()
            else:
                logger.warning(f"No valid response text for item {idx}. Finish reason: {candidate.finish_reason}")
                out_dict[out_column] = "C"  # Default to NOT_ATTEMPTED for safety-filtered responses
        else:
            logger.warning(f"No candidates in response for item {idx}")
            out_dict[out_column] = "C"
    except Exception as e:
        logger.error(f"Error processing item {idx} with Gemini after retries: {str(e)}")
        out_dict[out_column] = f"Error: {str(e)}"
    
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


async def main():
    from dotenv import load_dotenv
    load_dotenv()
    
    # Configure logging
    logger.add("evaluation.log", rotation="10 MB")
    
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
    filename = "/home/slurm/hoanganh/search_agent/data/frames_failed.parquet"
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
    df2 = evaluated_ds.to_pandas()
    print("num correct: ", sum(df2['is_correct'] == "A"))
    print("Accuracy: ", sum(df2['is_correct'] == "A") / len(df))
    df['is_correct'] = df2['is_correct'] == "A"
    df.to_parquet(filename)
    logger.info("Saved results locally to evaluated_results.parquet")


if __name__ == "__main__":
    asyncio.run(main())
from typing import List 
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI
import json
import os

client = OpenAI(api_key="replace with key")
def headline_rating(headlines: List[str], stock: str,) -> int:
    """
    headline: str of headline
    stock: stock of relevance
    output: integer from -5 to 5 indicating whether each headline is relevant to the stock
    """ 
    system_prompt = f"""You are analyzing the impact of news headlines on {stock}'s stock price within the next day.  
            Return a json object of {{query": integer}} from -5 to 5, inclusive, representing the expected price movement:  

            - **Positive values (1 to 5):** The headline is likely to cause an **increase** in {stock}'s price.  
            - **Negative values (-1 to -5):** The headline is likely to cause a **decrease** in {stock}'s price.  
            - **Higher absolute values (e.g., ±5):** Indicate **strong confidence** in the impact.  
            - **Lower absolute values (e.g., ±1):** Indicate **weaker confidence** in the impact.  
            - **Zero (0):** The headline is neutral or has **no effect** on {stock}'s price.  
            - try to pick up on anything that might be relevant to the stock price, no matter how small, for example if something would effect a supplier
            negatively that could effect the stock price negatively
            ### Important considerations:
            - **Not all headlines are equally important.** If some have more impact than others, reflect this in your score.  
            - **Only return json object of query and integer**  
            - **No preamble, no explanations.**  
            EXAMPLE JSON OUTPUT: "{{"query": 3}}" or "{{"query": -2}}"
        """
    user_prompt = "||".join(headlines)
    messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]
    
    llm_response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=messages,
        response_format={
            'type': 'json_object'
        },
        temperature = 0.9   
    )
    llm_response = json.loads(llm_response.choices[0].message.content)

    try:
        if abs(llm_response['query']) > 5:
            return 5
        elif abs(llm_response['query']) < -5:
            return -5
        return llm_response['query']
    except:
        print(f"Error: {e}")
        return 0

def relevance_extractor(headlines: List[str], sector: str) -> List[bool]:
    """
    headline: list of headlines
    sector: sector of relevance
    output: booleans indicating whether headline is relevant to the sector
    """ 
    system_prompt = f"""User will a headline. Assess whether each headline is relevant to determining if a stock in {sector} will rise or decrease. 
        Return a json object with key "query" and value boolean, indicating its relevance (True means relevant, False means irrelevent). Try to pick up on any small details that might be relevant like
        a headline about supply of a sector or any information that might affect the stock price. Key should be 'query' and value should be a single boolean.
        Make sure booleans are True or False, not true or false.
        """
    user_prompt = "||".join(headlines)
    messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]
    llm_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={
            'type': 'json_object'
        }
    )
    try:
        llm_response = json.loads(llm_response.choices[0].message.content)
        if type(llm_response['query']) is not bool:
            print("wrong data type returned")
            return False
        return llm_response['query']
    except (ValidationError, ValueError) as e:
        print(f"Error: {e} at {headlines}")
        return False
from typing import List
from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai.chat_models.base import BaseChatOpenAI

def relevance_extractor(headlines: List[str], sector: str) -> List[bool]:
    """
    headline: list of headlines
    sector: sector of relevance
    output: list of booleans indicating whether each headline is relevant to the sector
    """ 
    llm = BaseChatOpenAI(
        model='deepseek-chat', 
        openai_api_key='xdd', 
        openai_api_base='https://api.deepseek.com',
        max_tokens=1024
    )
    class Query(BaseModel):
        query: List[bool] = Field(
            description="""List of booleans indicating whether each headline is relevant to determining 
                        if a stock in the specified sector will rise or decrease. Treat || as delimiter"""
        )
    
    length = len(headlines)
    parser = JsonOutputParser(pydantic_object=Query)

    prompt = PromptTemplate(
        template="""Assess whether each headline is relevant to determining if a stock in {sector} will rise or decrease. 
        Return a JSON list of booleans, one for each headline, indicating its relevance. Output list should be of length {length}
        {format_instructions}

        Headlines:
        {headlines}
        """,
        input_variables= ["sector", "headlines", "length"],
        partial_variables={"format_instructions": parser.get_format_instructions(), },
    )
    
    chain = prompt | llm | parser
    llm_response = chain.invoke({"sector": sector,"headlines": "||".join(headlines), "length": length})
    try:
        if len(llm_response['query']) != length:
            raise ValueError("output list size does not match the number of headlines")
        return llm_response['query']
    except (ValidationError, ValueError) as e:
        print(f"Error: {e}")
        return [False] * length 
    

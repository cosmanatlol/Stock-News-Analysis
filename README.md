# Utilizing News in the Stock Market
## Multi-Model Sentiment Analysis with Technical Integration for Short-Term Stock Price Prediction
This is analysis of news headlines and how it relates to the stock market with predictions based on headlines and technical indicators
## Conda Environment
To download environment run the following:
```
conda env create -f environment.yml
conda activate stock-news-env
```
## How to use
### Scraper
scraper folder includes scrap.py which is how all the headline data is gathered from several news sources and test.ipnb which goes through how to use the scraper to generate your own data. 

### Headline Analysis
headline_analyiss folder includes Filter.py which is clustering similar articles together. llm_models which includes a relevancy filter going through Chat GPT and a scoring for the headline which also goes through Chat GPT. In order to run the LLMS you must replace the following line with your OpenAI API Key in llm_models.py:
```
client = OpenAI(api_key="replace with key")
```
and lastly there is rating_pipeline which puts together all the filters together and returns a score based on a stock. example.ipynb shows how to use the pipeline.

### Technical Analysis
tech_analysis folder includes ...

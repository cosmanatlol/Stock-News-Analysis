## Stock-News-Analysis
This is analysis of headlines and how it relates to a larger portfolio in the stock market with all tools given.
# Conda Environment
To download environment run "conda env create -f environment.yml" and then run "conda activate stock-news-env". 
# How to use
Stock selection folder contains code on how stocks were selected initially. Scraper folder contains code on how to scrape data with test.ipynb as an example and scrap.py being source code. headline_analysis folder contains the pipeline to create predictions based on headlines including all three layers to generate a prediction and example.ipyb as an example of how to use the pipeline. Example takes approximately 20 hours to run. You must generate your own OPENAI key and put it into llm_models. Tech analysis includes files on the technical analysis and the overall final model that was used for analysis. 

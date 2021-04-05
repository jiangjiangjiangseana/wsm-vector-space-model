# Vector Space Model 

Web Mining and Search course practice project 

Establish vector space model in practice to retrieve relevence top 10 document with user query. 

## How to run the code

### Initiate environment

` pip install -r requirements.txt`

### Run

`python main.py --query {query_str}`

Notice : 
    
- query_str should be input within " " .

- queries should be seperated by space .

Example :

`python main.py --query "Trump Biden Taiwan China"`

Output : top 10 news ID with scores

- Term Frequency Weighting + Cosine Similarity

- Term Frequency Weighting + Euclidean Distance

- TF-IDF Weighting + Cosine Similarity

- TF-IDF Weighting + Euclidean Distance

- Relevence Feedback - TF-IDF + Cosine Similarity

The code runs for a long time on my computer.
It costs more than 30 minutes :(

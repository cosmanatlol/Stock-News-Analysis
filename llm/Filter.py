
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

try:
    nltk.data.find('stopwords')
    nltk.data.find('punkt_tab')
    nltk.data.find('wordnet')
    nltk.data.find('punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('punkt')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# normalizing text - punctuation, capitalization, lemmatization
def normalize_text(text):
    lowered = text.lower()
    nopunc = re.sub(r'[^\w\s]', '', lowered)
    tokens = word_tokenize(nopunc)
    words = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(words)


# clustering titles
def cluster_similar_titles(normalized_titles, similarity_threshold=0.7):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(normalized_titles)

    # hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters = None,
        distance_threshold = 1 - similarity_threshold,
        metric = 'cosine',
        linkage = 'average'
    )
    clusters = clustering.fit_predict(embeddings)
    
    return clusters


# applying filtering
def filtering(titlesdf, threshold = 0.7):
    """
    titlesdf - pandas dataframe containing "headlines", "date", and "source"
    threshold - similarity threshold cutoff value (0 = least similar, 1 = most similar)
    """
    titlesdf['headline_norm'] = titlesdf['headline'].apply(normalize_text)
    titlesdf["cluster_id"] = cluster_similar_titles(titlesdf['headline_norm'], threshold)

    titlesdf = titlesdf.sort_values('date').drop_duplicates(subset='cluster_id', keep='first')
    return titlesdf.drop(columns=['headline_norm', 'cluster_id'])



# titlesdf = pd.read_csv("data.csv")
# print(filtering(titlesdf[:100]))


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering


# normalizing text - punctuation, capitalization, lemmatization
def normalize_text(text):
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    nltk.download('wordnet')

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

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
    
    clustered_titles = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in clustered_titles:
            clustered_titles[cluster_id] = []
        clustered_titles[cluster_id].append(normalized_titles[i])
    
    return clustered_titles, x


# applying filtering
def filtering(titles, threshold = 0.7):
    titlesn = titles.apply(normalize_text)
    titlesc = cluster_similar_titles(titlesn, threshold)

    return titlesc

"""
This is a simple application for sentence embeddings: clustering

Sentences are mapped to sentence embeddings and then agglomerative clustering with a threshold is applied.
"""

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Corpus with example sentences
""" corpus = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "A man is eating pasta.",
    "The girl is carrying a baby.",
    "The baby is carried by the woman",
    "A man is riding a horse.",
    "A man is riding a white horse on an enclosed ground.",
    "A monkey is playing drums.",
    "Someone in a gorilla costume is playing a set of drums.",
    "A cheetah is running behind its prey.",
    "A cheetah chases prey on across a field.",
] """

def weight_first_words(embeddings, weight=2.0):
    weighted_embeddings = embeddings
    print(weighted_embeddings)
    for sent in weighted_embeddings:
        if len(sent) >= 2:
            sent[:2] *= weight
    return weighted_embeddings


df = pd.read_excel("Firma İsimleri.xlsx")

print(df.head())
print(df["Mevcut Firma Adı"].values)
corpus = df.loc[:, "Mevcut Firma Adı"].values
corpus_embeddings = embedder.encode(corpus)

# Some models don't automatically normalize the embeddings, in which case you should normalize the embeddings:
corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

corpus_embeddings = weight_first_words(corpus_embeddings, 6.0)
# Perform agglomerative clustering
clustering_model = AgglomerativeClustering(
    n_clusters=None, distance_threshold=1.2
)  # , affinity='cosine', linkage='average', distance_threshold=0.4)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = {}
for sentence_id, cluster_id in enumerate(cluster_assignment):
    if cluster_id not in clustered_sentences:
        clustered_sentences[cluster_id] = []

    clustered_sentences[cluster_id].append(corpus[sentence_id])

for i, cluster in clustered_sentences.items():
    print("Cluster ", i + 1)
    print(cluster)
    print("")

text_dict = {}
for i, cluster in clustered_sentences.items():
    print("\nCluster {}, #{} Elements ".format(i + 1, len(cluster)))
    for sent in set(cluster):
        if sent not in text_dict.keys():
            text_dict[sent] = i

df["Olması Gereken Firma Adı"] = df["Mevcut Firma Adı"].map(text_dict)

df.to_excel("out2_slow.xlsx")
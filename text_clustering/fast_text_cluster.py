"""
This is a more complex example on performing clustering on large scale dataset.

This examples find in a large set of sentences local communities, i.e., groups of sentences that are highly
similar. You can freely configure the threshold what is considered as similar. A high threshold will
only find extremely similar sentences, a lower threshold will find more sentence that are less similar.

A second parameter is 'min_community_size': Only communities with at least a certain number of sentences will be returned.

The method for finding the communities is extremely fast, for clustering 50k sentences it requires only 5 seconds (plus embedding comuptation).

In this example, we download a large set of questions from Quora and then find similar questions in this set.
"""

from sentence_transformers import SentenceTransformer, util
import time
import pandas as pd


def weight_first_words(embeddings, weight=2.0):
    weighted_embeddings = embeddings.clone()
    print(weighted_embeddings)
    for sent in weighted_embeddings:
        if len(sent) >= 2:
            sent[:2] *= weight
    return weighted_embeddings

# Model for computing sentence embeddings. We use one trained for similar questions detection
model = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_excel("Firma İsimleri.xlsx")

print(df.head())
print(df["Mevcut Firma Adı"].values)
corpus = df.loc[:, "Mevcut Firma Adı"].values
print(corpus)



print("Encode the corpus. This might take a while")
corpus_embeddings = model.encode(corpus, batch_size=64, show_progress_bar=True, convert_to_tensor=True)

corpus_embeddings = weight_first_words(corpus_embeddings, 0.6)

print("Start clustering")
start_time = time.time()

# Two parameters to tune:
# min_cluster_size: Only consider cluster that have at least 25 elements
# threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
clusters = util.community_detection(corpus_embeddings, threshold=0.7)

print("Clustering done after {:.2f} sec".format(time.time() - start_time))

# Print for all clusters the top 3 and bottom 3 elements
for i, cluster in enumerate(clusters):
    print("\nCluster {}, #{} Elements ".format(i + 1, len(cluster)))
    for sentence_id in cluster:
        print("\t", corpus[sentence_id])
    print("\t", "...")
    for sentence_id in cluster:
        print("\t", corpus[sentence_id])

text_dict = {}
for i, cluster in enumerate(clusters):
    print("\nCluster {}, #{} Elements ".format(i + 1, len(cluster)))
    for sentence_id in set(cluster):
        sent = corpus[sentence_id]
        if sent not in text_dict.keys():
            text_dict[sent] = i

print(text_dict)

df["Olması Gereken Firma Adı"] = df["Mevcut Firma Adı"].map(text_dict)

df.to_excel("out2.xlsx")
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/movie_dataset.csv")
print(df.info())
# Helper functions
# Get the title of the movie from its index in dataframe
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]
# Get the index of the matched movie title
def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]
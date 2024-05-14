import pandas as pd
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    return filtered_tokens

def perform_lda(texts, num_topics=14):
    processed_texts = [preprocess_text(text) for text in texts]
    dictionary = corpora.Dictionary(processed_texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=15, random_state=42)
    return lda_model, corpus, dictionary

def get_topic_distribution(lda_model, corpus):
    topic_distribution = [0] * lda_model.num_topics
    for document in corpus:
        for topic, prob in lda_model.get_document_topics(document):
            topic_distribution[topic] += prob
    return topic_distribution

def print_ranked_topics(lda_model, topic_distribution):
    ranked_topics = sorted(list(enumerate(topic_distribution)), key=lambda x: x[1], reverse=True)
    print("\nRanked Topics from Most to Least Mentioned:")
    for idx, weight in ranked_topics:
        print(f"Topic {idx} (Total Weight: {weight:.4f}): {lda_model.print_topic(idx, 5)}")

def main():
    print("Loading data...")
    df = pd.read_csv('negative_comments.csv', usecols=[0], header=None)
    negative_comments = df[0].dropna().tolist()

    print("Performing LDA to discover topics...")
    lda_model, corpus, dictionary = perform_lda(negative_comments, num_topics=14)

    topic_distribution = get_topic_distribution(lda_model, corpus)
    print_ranked_topics(lda_model, topic_distribution)

if __name__ == "__main__":
    main()

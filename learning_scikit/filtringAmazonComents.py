import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

class Sentiment:
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"

class Review:
    def __init__(self, text, rating):
        self.text = text
        self.rating = rating
        self.sentiment = self.get_sentiment()

    def get_sentiment(self):
            if self.rating <= 2:
                return Sentiment.NEGATIVE
            elif self.rating == 3:
                return Sentiment.NEUTRAL
            else:
                return Sentiment.POSITIVE

file_name = './data/Books_small.json'

reviews = []

with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'], review['overall']))

training, test = train_test_split(reviews, test_size=0.33, random_state=42)
train_x = [x.text for x in training]
train_y = [x.sentiment for x in training]

test_x = [x.text for x in test]
test_y = [x.sentiment for x in test]

vectorizer = CountVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)


print(train_x_vectors[0])
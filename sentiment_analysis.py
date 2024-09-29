import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import seaborn as sns
from textblob import Word, TextBlob
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from warnings import filterwarnings
from datetime import datetime


filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

dataset = pd.read_excel("./data/amazon.xlsx")
df = dataset.copy()

print(df.head())

# 1. Text Preprocessing

# a. Change all letters to lower case.

df["Review"] = df["Review"].str.lower()

# b. Remove the punctuation marks.

df["Review"] = df["Review"].str.replace(r"[^\w\s]", "", regex=True)

# c. Remove the numerical expressions in the comments.

df["Review"] = df["Review"].str.replace(r"/d", "", regex=True)

# d. Remove non-informative words (stopwords) from the data.

nltk.download("stopwords")
stop_words = stopwords.words("english")

df["Review"] = df["Review"].apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))

# e. Remove words with less than 50 occurrences from the data.

tf_words = pd.Series(" ".join(df["Review"]).split()).value_counts()

drops = tf_words[tf_words < 50]

df["Review"] = df["Review"].apply(lambda x: " ".join(x for x in str(x).split() if x not in drops))

# f. Lemmatization

lem = WordNetLemmatizer()

df["Review"] = df["Review"].apply(lambda x: " ".join(lem.lemmatize(x, pos="v") for x in str(x).split()))

# Text Visualization
 
# Barplot

tf = pd.Series(" ".join(df["Review"]).split()).value_counts()

tf = pd.DataFrame(tf).reset_index()

tf.columns = ["words", "tf"]


tf_500 = tf[tf["tf"] > 500]

plot_data = {"words": list(tf_500["words"]),
             "tf": list(tf_500["tf"])}

now = datetime.now()

sns.barplot(x="tf", y="words", data=plot_data)
plt.show()

# WordCloud

text = " ".join(df["Review"])

wordcloud = WordCloud(background_color="white").generate(text)

plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# 3. Sentiment Analysis

sia = SentimentIntensityAnalyzer()

df["Review"].iloc[10]

sia.polarity_scores("The film was awesome")

df["polarity_score"] = df["Review"].apply(lambda x: sia.polarity_scores(x)["compound"])

df["sentiment_label"] = df["polarity_score"].apply(lambda x: "pos" if x > 0 else "neg")

# 4. Preparing for Machine Learning

# a. Identify our dependent and independent variables and separate the data as train test.

y = df["sentiment_label"]
X = df["Review"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

tf_idf_word_vectorizer = TfidfVectorizer()

X_train_tf_idf_word = tf_idf_word_vectorizer.fit_transform(X_train)
X_test_tf_idf_word = tf_idf_word_vectorizer.fit_transform(X_test)

# b. Modeling (Logistic Regression)

log_model = LogisticRegression().fit(X_train_tf_idf_word, y_train)

y_pred = log_model.predict(X_test_tf_idf_word)

print(classification_report(y_pred, y_test))

print(cross_val_score(log_model, X_test_tf_idf_word, y_test, cv=5).mean())

# c. Randomly selecting comments from the data and asking the model

sample = pd.Series(df["Review"].sample(1).values) # sample = pd.Series("Bad curtain")
yeni_yorum = CountVectorizer().fit(X_train).transform(sample)
y_pred = log_model.predict(yeni_yorum)
print(f"Review: {sample[0]}\nPredict: {y_pred}")


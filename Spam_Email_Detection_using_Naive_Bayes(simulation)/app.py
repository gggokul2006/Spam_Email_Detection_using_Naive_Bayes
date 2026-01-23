from flask import Flask, render_template, request
import pandas as pd
import re
import nltk
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)


nltk.download('stopwords')

app = Flask(__name__)


data = pd.read_csv("spam.csv", encoding="latin-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'message']


ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

data['clean'] = data['message'].apply(preprocess)


tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data['clean']).toarray()
y = data['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


acc  = round(accuracy_score(y_test, y_pred) * 100, 2)
prec = round(precision_score(y_test, y_pred) * 100, 2)
rec  = round(recall_score(y_test, y_pred) * 100, 2)
f1   = round(f1_score(y_test, y_pred) * 100, 2)


cm = confusion_matrix(y_test, y_pred).tolist()


fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = round(auc(fpr, tpr), 3)


fpr = [round(x, 3) for x in fpr]
tpr = [round(x, 3) for x in tpr]


@app.route("/", methods=["GET", "POST"])
def dashboard():
    prediction = ""
    msg = ""

    if request.method == "POST":
        msg = request.form["message"]
        clean = preprocess(msg)
        vec = tfidf.transform([clean]).toarray()
        result = model.predict(vec)
        prediction = "SPAM" if result[0] == 1 else "NOT SPAM"

    return render_template(
        "dashboard.html",
        acc=acc,
        prec=prec,
        rec=rec,
        f1=f1,
        cm=cm,
        fpr=fpr,
        tpr=tpr,
        auc=roc_auc,
        prediction=prediction,
        msg=msg
    )


if __name__ == "__main__":
    app.run(debug=True)


from flask import Flask, render_template, request
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

ps = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    words = [ps.stem(w) for w in words]
    return " ".join(words)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    email = request.form['email']
    email = preprocess_text(email)

    vector = vectorizer.transform([email])
    prediction = model.predict(vector)

    if prediction[0] == 1:
        result = "Spam Email 🚨"
    else:
        result = "Not Spam Email ✅"

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)
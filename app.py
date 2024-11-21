import re
import io
import statistics
import base64

from transformers import pipeline
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import markdown as md

import bot

classifier = pipeline("sentiment-analysis")
app = Flask(__name__)


def shift(value, weight):
    precision = value * weight
    result = re.search(r"\d\d\..+", str(precision)).group()
    return float(result)


def sentiment_status(text):
    sentiment = classifier(text)[0]
    score = shift(sentiment["score"], 10000)
    if sentiment["label"] == "POSITIVE":
        positive_scores.append(score)
        negative_scores.append(100 - score)
    else:
        negative_scores.append(score)
        positive_scores.append(100 - score)


queries = set()
dialogues = []

positive_scores = []
negative_scores = []
CRISIS_THRESHOLD = 97.5


def emotion_values():
    ngtve = statistics.mean(negative_scores)
    pstve = statistics.mean(positive_scores)
    return [ngtve, pstve]


@app.route("/")
def index():
    piechart_img = None
    if "prompt" in request.args:
        query = request.args["prompt"]
        if query not in queries:
            sentiment_status(query)
            sizes = emotion_values()
            if sizes[0] >= CRISIS_THRESHOLD:
                return render_template("crisis.html")

            plt.switch_backend('Agg')
            plt.figure(figsize=(6, 6))
            plt.pie(sizes, colors=["red", "blue"],
                    autopct='%1.1f%%', shadow=True, startangle=140)
            plt.axis('equal')

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plt.close()

            piechart_img = base64.b64encode(buf.getvalue()).decode('utf8')
            response = bot.respond_to(query)
            response = md.markdown(response,
                                   extensions=['extra', 'codehilite', 'fenced_code', 'tables', 'nl2br', 'sane_lists',
                                               'smarty', 'toc', 'wikilinks'])
            dialogues.append({"role": "user", "message": query})
            dialogues.append({"role": "bot", "message": response})
            queries.add(query)

    return render_template("index.html", chats=dialogues, piechart_img=piechart_img)


if __name__ == "__main__":
    app.run(debug=True)

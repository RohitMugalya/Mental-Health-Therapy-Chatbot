import os
from pathlib import Path
import re
import statistics

from transformers import pipeline
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import markdown as md

import bot

PIE_CHART = Path(r"./static/pie_chart.png")
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
    pie_chart = PIE_CHART if PIE_CHART.exists() else None
    if "prompt" in request.args:
        query = request.args["prompt"]
        if query not in queries:
            sentiment_status(query)
            sizes = emotion_values()
            if sizes[0] >= CRISIS_THRESHOLD:
                return render_template("crisis.html")

            pie_chart = PIE_CHART
            plt.switch_backend('Agg')
            plt.figure(figsize=(6, 6))
            plt.pie(sizes, colors=["red", "blue"],
                    autopct='%1.1f%%', shadow=True, startangle=140)
            plt.axis('equal')

            plt.savefig(pie_chart, bbox_inches='tight')
            plt.close()

            response = bot.respond_to(query)
            response = md.markdown(response,
                                   extensions=['extra', 'codehilite', 'fenced_code', 'tables', 'nl2br', 'sane_lists',
                                               'smarty', 'toc', 'wikilinks'])
            dialogues.append({"role": "user", "message": query})
            dialogues.append({"role": "bot", "message": response})
            queries.add(query)

    return render_template("index.html", chats=dialogues, pie_chart=pie_chart)


if __name__ == "__main__":
    app.run(debug=True)
    if PIE_CHART.exists():
        os.remove(PIE_CHART)

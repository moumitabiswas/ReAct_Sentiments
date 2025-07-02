import pandas as pd
from textblob import TextBlob
import dspy


def sentiments(text):
    return TextBlob(text).sentiment.polarity


tool_sentiments=dspy.Tool(
       name="Sentiment Analysis and Score",
       desc="Detect the sentiments of the text",
       func=sentiments)



class GenerateSentiment(dspy.Signature):
    """Detect the sentiments."""
    text = dspy.InputField(desc="contain relevant facts and comments")
    # question = dspy.InputField()
    sentiments = dspy.OutputField(desc="In one word either Positive, Negative or Neutral")


class MyReAct(dspy.Module):
    def __init__(self, tools):
        super().__init__()
        self.ra_module = dspy.ReAct(GenerateSentiment, tools=tools)

    def forward(self, text):
        response = self.ra_module(text=text,tools=tool_sentiments)
        return response



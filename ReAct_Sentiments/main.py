import dspy
import sys
import os
import openai
import json
from dspy.evaluate.evaluate import Evaluate
from dspy import teleprompt
import pandas as pd
import os
from config_llm import llm
from data_load.data_load import dataload, inputtext
from react import sentiments
from react import tool_sentiments, MyReAct

print("Loading the libraries successful")

df = dataload(r"")
print(f"The input data :{df.head(1)}")

inp_text= inputtext(df)

print(f"The input text :{inp_text[0:2]}")

test_text=inp_text[8]
my_react_module = MyReAct(tools=[tool_sentiments])

print(f"The input review is: {test_text}")

print(f"The ReAct Agent how thinks and take decisions in steps: {my_react_module(test_text)}")

""" END """
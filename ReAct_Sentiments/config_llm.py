
import os 
import dspy
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

llm=dspy.LM(model='openai/gpt-4o', api_key=OPENAI_API_KEY, max_tokens=1000, temperature=0.1)
dspy.configure(lm=llm,adapter=dspy.TwoStepAdapter(llm))
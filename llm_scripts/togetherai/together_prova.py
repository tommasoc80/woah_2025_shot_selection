from together import Together
import pandas as pd

SYSTEM_PROMPT = """You are an AI expert in text classification and content moderation. 
You will read a text related to Brexit. Does the text contain hate speech?
Answer only with 0 for NOT and 1 for YES.
Do not write an introduction or summary. Classify always the text.
Here are twenty examples:
{'text': '".<user> We'll get exactly the same immigration system. #Brexit told a lie. Gullible fell for it. <user> <user>"', 'label': 0},
{'text': '"Retweeted MUSLIM PATROL (<user>:  #Brexit was not about racism Yeh Right <url> <url>"', 'label': 0},        
{'text': '"RT <user>: Calais Muslim Migrants Block Roads, Destroy Cars, Shout 'Fuck UK' #BanIslam #StopIslam #Brexit #VoteLeave <url>"', 'label': 1},
{'text': 'Y is everyone affraid to send this scum of Islam packing <url>', 'label': 1},
{'text': 'RT <user>: amanpour's a pro- refujihadi rapefugee dhimmi. See how that works? #MakeBritainGreatAgain <url>', 'label': 1},
{'text': 'misogynistic xenophobic white supremacist terrorist encouraging cowards don't want what is right they just want their way. #Brexit', 'label': 0},    
{'text': 'Michael Gove sets up new law to thwart ‘rogue’ EU courts &amp; deport foreign criminals <url> #Brexit <url>', 'label': 1},
{'text': 'Mother F*CK the Council on Foreign Relations and anything they may have to say. 💩😬💩😬💩 #EUref #Brexit', 'label': 0},
{'text': 'Because immigration is mainly a business matter... <user> in <user> #brexit #immigration #BorisJohnson <url>', 'label': 0},
{'text': 'so equal pay for women would have stopped fears of immigration #brexit #zimmer conflation', 'label': 0},        
{'text': 'A significant day to be discussing foreign policy and #Brexit with former Secretary of State Henry Kissinger. <url>', 'label': 0},       
{'text': '"Oh great, that should make Europe's slide into Islam just that much easier. why don't you just shoot yourselves now. <url>"', 'label': 1}, 
{'text': 'Welcome British Refugees #brexit #EUref <url>', 'label': 0},
{'text': 'Educate yourself. IS hate anyone who doesn't share their beliefs. Leaving the EU won't stop terrorist attacks <url>', 'label': 0},       
{'text': 'RT <user>: #IslamIsTheProblem. Islam is a disgrace! This is not a religion it's a cult! #banMuslims  <url>', 'label': 1},       
{'text': '"After #Brexit #UK can deport #radical #Islam #muslim ""#clerics"" (AKA #isis recruiters #terrorists) #EU ""#humanright "" except for natives"', 'label': 1},        
{'text': 'F*** Obama and nobody listen to anymore f*** that Muslim <url>', 'label': 1},        
{'text': '"#Brexit Facts about Pakistanis 1) Everyone hates them 2) Pakis only hate Hindus 3) Pakis dogs of Arabians"', 'label': 1},
{'text': '"#Brexit will impact so many different things stock market, trade, employment, immigration etc Thursday is a big day <url>"', 'label': 0},       
{'text': 'No more Islam #HENK <url>', 'label': 1}  

Now classify whether the following text is sexist:'''{text}'''
Respond only with the valid JSON format below:
{{'text': '{text}', 'label': …}}
"""

TOGETHER_API_KEY = "29694f02806af48449982236bf9ad5b4e0daf9951c4382b9b4b0d6002094bdd0"
client = Together(api_key= TOGETHER_API_KEY)


def run_llm(model, text_data):


  stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "What are the top 3 things to do in New York?"}],
    stream=True,
  )

  for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)


if __name__ == '__main__':

  df = pd.read_csv("/scratch/p281734/Brexit_hard_label_test.csv", sep=",", header=0)
  model = "meta-llama/Meta-Llama-3-70B-Instruct"

  messages = df["text"]
  run_llm(model, messages)






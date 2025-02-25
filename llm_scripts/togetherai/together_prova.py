from together import Together
import pandas as pd
from pydantic import Field, BaseModel, ValidationError
from typing import Any, Optional, Dict, List, Literal

SYSTEM_PROMPT = "You are an AI expert in text classification and content moderation."
INSTRUCTION_PREAMBLE = """You will read a text. Is the text sexist?.
Answer only with 0 for NOT and 1 for YES.
Do not write an introduction or summary. Classify always the text.
Here are twenty examples:
```
{'text': 'Shouldn’t he be preparing his last will & testament instead of tweeting stupid shit and further embarrassing himself and eroding his reputation and legacy?', 'label': 0}},
{'text': 'Not in the case of Muslim woman', 'label': 0}
{'text': 'Sub-Saharan Africans (even half-breeds) don't exactly bring up images of high quality manufacturing in my mind. What about the rest of you.', 'label': 0}c
{'text': '"And, her daddy was Chief Spreading Bull"', 'label': 0}
{'text': 'It's a 4chan post it probably 404'd', 'label': 0}
{'text': 'Chads prom night ended as a foursome surrounded by dozens of Stacie’s while incels prom night ends in their jaw slipping through the noose', label': 0}
{'text': 'Ok 5’8 very attractive guy vs 6’5 ugly guy the 5’8 guy would get the girls.', 'label': 0}
{'text': 'YouTube removes videos from conspiracy theorist Alex Jones: Infowars website! #CNN is behind this attack! [URL]', 'label': 0}
{'text': 'we don't know if that is ZUMA. we are not sure about that yet.', 'label': 0}
{'text': '""And access to a single market with lucrative free trade arrangements", the nine-year-old cooed as she sold her kittens to the friendly Tom Ka Gai pushcart vendor on the corner."', 'label': 0}}
{'text': 'Here she is the breeding machine No thoughts for how her kids will work rest play eh Africa the same Muslims the same They All sicken me', 'label': 1}     
{'text': '"Not true. TRP is 100% beneficial to her, provided she is the least bit fuckable. um...no, no it's not."', 'label': 1}
{'text': '"If you read the newspaper one day that some crazy bitch murdered her husband, and you recognize a dumped first date in the mugshot, you will thank me."', 'label': 1}
{'text': 'Fucking disgusting you can see the kid trying to hold on single motherhood is a plague avoid that shit AT ALL COST', 'label': 1}
{'text': 'gym thots are fishing with really wide nets-- tread carefully.', 'label': 1}
{'text': '"1&7. Feminists are the worst. For example, that nasty bitch that made bread with her yeast infection riddled coochie goo and threatened to give to her family that she didnt like at Thanksgiving dinner. Sick demented shit!"', 'label': 1}
{'text': '"If women can think the air conditioning can oppress them, there is no limit to the shit they will make up."', 'label': 1}}
{'text': '"Seriously. All women think about is status and sex, and then crotch spawns after they shit their first one out."}. 'label': 1}
{'text': 'Unattractive women don't get bred out and pass on their defects to their offspring.', 'label': 1}
{'text': '"They see but theyll say some bs like"" i know my worth"". Fds women are so sub par that even tbe 6/10 white women on my tinder mog the entre sub forum to oblivion.", 'label': 1}
```
"""

INSTRUCTION_TEMPLATE = """Now classify whether the following text is sexist:'''{text}'''
Respond only with the valid JSON format below:
{{'text': '{text}', 'label': }}
"""

TOGETHER_API_KEY = "29694f02806af48449982236bf9ad5b4e0daf9951c4382b9b4b0d6002094bdd0"
client = Together(api_key= TOGETHER_API_KEY)


def conversations_from_example(example_text):
  return [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
      "role": "user",
      "content": INSTRUCTION_PREAMBLE
                 + INSTRUCTION_TEMPLATE.format(text=example_text),
    },
  ]


def run_llm(model, messages):

    responses = []
    messages_list = [conversations_from_example(text) for text in messages]
    #print(messages_list)

    for entry in messages_list:
        response = client.chat.completions.create(
                    model=model,
                    messages=entry,
                    temperature=0.0,
                    max_tokens=500,
                    )

        if str(response.choices[0].message.content).startswith("{"):
            # label_extracted = re.search("\'label\': (\w+)", response)
            keyword = "\'label\':"
            before_keyword, keyword, after_keyword = str(response.choices[0].message.content).partition(keyword)
            #        print(after_keyword.replace("}]","").replace("}", ""))
            clean_answer = after_keyword.replace("}]", "").replace("}", "").replace("\"", "").replace("\n", "").replace(
                "]", "")
            responses.append(clean_answer)
        else:
            #        print("Refused")
            responses.append("Refused")

    return responses
""""




  stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "What are the top 3 things to do in New York?"}],
    stream=True,
  )

  for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)


    messages = []
    if system_prompt:
      messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_prompt})

    

    return response.choices[0].message.content
"""

if __name__ == '__main__':

#    df = pd.read_csv("/scratch/p281734/Brexit_hard_label_test.csv", sep=",", header=0)
    df = pd.read_csv("prova.csv", sep=",", header=0)
    model = "Qwen/Qwen2.5-72B-Instruct-Turbo"
# meta-llama/Meta-Llama-3-70B-Instruct-Turbo
    messages = df["text"]

    responses = run_llm(model, messages)
    df['model_answer'] = responses
#    df.to_csv('/scratch/p281734/brexit_qwen_72_ambiguous_first.csv', index=False)
    df.to_csv('prova_qwen_72.csv', index=False)





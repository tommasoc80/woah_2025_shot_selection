from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch
import pandas as pd

model = "meta-llama/Llama-3.2-3B-Instruct"
#model = "~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(model)

df = pd.read_csv('/scratch/p281734/ghc_test.tsv', sep='\t', header=0)
#df = pd.read_csv('/scratch/p281734/prova.csv', sep=',', header=0)

responses = []

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

for index, row in df.iterrows():
    text = row['text']
    messages = [
        {"role": "system", "content": "You are an AI expert in text classification and content moderation."},
        {"role": "user", "content": """
        You will read a text. Does the text contain hate speech?
        Answer only with 0 for NOT and 1 for YES.
        Do not write an introduction or summary. Classify always the text.
        Here are twenty examples:
        {{ 'response': [
            {{'text': 'He never delves into what SHOULD be done about the problem and always seems to stay away from actual politics and yet his views are censored. There is no socially exceptable way to criticize blacks from full on hang them all to they should be treated the same way under the law(his view)',
            'label': 0}},
            {{'text': '"This tribalism is the problem with libertarianism – if you allow liberty, people will use it to synthesize smaller ingroups within the larger group in order to dominate the detribalized majority. - Jim"',
            'label': 0}},        
            {{'text': '"Didn't she get benefits to attend college as a ""Native American:?"',
            'label': 0}},
            {{'text': '"Washington DC is the first place to have armed guards at every door and the last to deserve them. If you piss us off that bad,you deserve to get shot . Remember  "',
            'label': 0}},
            {{'text': 'Face it. America is a shit hole country that doesn't even care enough about its own children to actually make efforts to keep them safe. You can fuck off now.',
            'label': 0}},
            {{'text': 'That is so vulgar I can hardly believe someone would willingly participate in that.',
            'label': 0}},                
            {{'text': 'It's frustrating to watch politicians who know next to nothing about a topic write law!',
            'label': 0}},
            {{'text': '"My favorite hypocrisy of Dorsey is he suspended Alex for ""incitement of violence"" while letting #rapeMelania trend after the election and doing nothing about it when brought to his attention. Burn Twatter!"',
            'label': 0}},
            {{'text': 'The Democaust or Demopocalypse (whichever) will be fun times.',
            'label': 0}},
            {{'text': '"The sooner DACA has died and ""Dreamers"" are shown the door, the soon Americans can be safe from the ""Dreamer"" crime rampage."',
            'label': 0}},        
            {{'text': '"Man, this delusion neckbeard has a wife who doesn't have a forehead...That cone skull ranks at least a tenhead.  Geez.https://www.thesun.co.uk/news/7148589/inside-churc..."',
            'label': 1}},        
            {{'text': 'IE: Most humans are fucking ignorant and arrogant AND most ETs are fucking amazingly honourable and wise.  See?  Balance.  :>',
            'label': 1}},
            {{'text': '"Dont we want hate speech laws? It highlights the problem, what am i missing?   Long before hate speech laws white opinions have been silenced, supressed and ostracised.   Ive been trying to get the state to prosecute me under the hate speech laws but they wont.  "',
            'label': 1}},       
            {{'text': '"the so called jews, are not jews at all, but are of CAIN. Western whites are literally the so called 10 lost tribes of ISRAEL. britian is thought to be the children of Joseph, Ephraim (europe) and the US is thought to be Mannaseh (cow pokes). Naturally the children of SATAN, CAINs father, wants to destroy the ISRAELITES the true CHOSEN"',
            'label': 1}},       
            {{'text': 'It's time to start requiring citizens to have skin in the game before they are allowed to vote',
            'label': 1}},        
            {{'text': '"And they continue to do so, R'abbi. It's as simple as that. (((The enemy))) wants to take over the world, and they want to kill you and me. Heretical rabbis they will burn at the stake, incidentally. So what will we do about it-?"',
            'label': 1}},        
            {{'text': 'That Time Telco Lobbyists Sent Me All Their Talking Points About Trying To Shift The Blame To Internet Companies https://www.techdirt.com/articles/20180829/1803074...',
            'label': 1}},
            {{'text': 'A church in Berlin was damaged after unknown attackers pelted stones at the building while shouting “Allahu Akbar!” https://www.infowars.com/berlin-attackers-hurl-sto...',
            'label': 1}},       
            {{'text': '#PervPastor copping feels at Aretha Franklin's funeral. Nothing but a bunch of thug . criminals #ArethaFranklinFuneral #FridayFeeling #SpeakFreely',
            'label': 1}},        
            {{'text': 'Who cares what John Dean says. The commies keep living in the past.',
            'label': 1}}        
                    ]}}
        Now classify the following text: '''{text}'''
        Respond only with the valid JSON format below:
        {{ 'response' : [
            {{'text': '{text}',
            'label': }}
                    ]}}

""".format(text=text)},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

#    terminators = [
#        pipeline.tokenizer.eos_token_id,
#        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
#        eos_token_id=terminators,
        do_sample=False,
        temperature=0,
)

#    print(outputs)
    response = outputs[0]["generated_text"][len(prompt):]
#    print(response)
#    responses.append(response)

    if str(response).startswith("{"):
        #label_extracted = re.search("\'label\': (\w+)", response)
        keyword = "\'label\':"
        before_keyword, keyword, after_keyword = str(response).partition(keyword)
#        print(after_keyword.replace("}]","").replace("}", ""))
        clean_answer = after_keyword.replace("}]","").replace("}", "").replace("\"","").replace("\n","").replace("]","")
        responses.append(clean_answer)
    else:
#        print("Refused")
        responses.append("Refused")


df['model_answer'] = responses
df.to_csv('/scratch/p281734/gab_llama3_difficult_first.csv', index=False)


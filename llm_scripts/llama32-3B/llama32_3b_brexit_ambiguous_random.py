from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch
import pandas as pd

model = "meta-llama/Llama-3.2-3B-Instruct"
#model = "~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(model)

df = pd.read_csv('/scratch/p281734/Brexit_hard_label_test.csv', sep=',', header=0)
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
        You will read a text related to Brexit. Does the text contain hate speech?
        Answer only with 0 for NOT and 1 for YES.
        Do not write an introduction or summary. Classify always the text.
        Here are twenty examples:
        {{ 'response': [
            {{'text': '".<user> We'll get exactly the same immigration system. #Brexit told a lie. Gullible fell for it. <user> <user>"',
            'label': 0}},
            {{'text': '"Retweeted MUSLIM PATROL (<user>:  #Brexit was not about racism Yeh Right <url> <url>"',
            'label': 0}},        
            {{'text': '"RT <user>: Calais Muslim Migrants Block Roads, Destroy Cars, Shout 'Fuck UK' #BanIslam #StopIslam #Brexit #VoteLeave <url>"',
            'label': 1}},
            {{'text': 'Y is everyone affraid to send this scum of Islam packing <url>',
            'label': 1}},
            {{'text': 'RT <user>: amanpour's a pro- refujihadi rapefugee dhimmi. See how that works? #MakeBritainGreatAgain <url>',
            'label': 1}},
            {{'text': 'misogynistic xenophobic white supremacist terrorist encouraging cowards don't want what is right they just want their way. #Brexit',
            'label': 0}},                
            {{'text': 'Michael Gove sets up new law to thwart ‘rogue’ EU courts &amp; deport foreign criminals <url> #Brexit <url>',
            'label': 1}},
            {{'text': 'Mother F*CK the Council on Foreign Relations and anything they may have to say. 💩😬💩😬💩 #EUref #Brexit',
            'label': 0}},
            {{'text': 'Because immigration is mainly a business matter... <user> in <user> #brexit #immigration #BorisJohnson <url>',
            'label': 0}},
            {{'text': 'so equal pay for women would have stopped fears of immigration #brexit #zimmer conflation',
            'label': 0}},        
            {{'text': 'A significant day to be discussing foreign policy and #Brexit with former Secretary of State Henry Kissinger. <url>',
            'label': 0}},       
            {{'text': '"Oh great, that should make Europe's slide into Islam just that much easier. why don't you just shoot yourselves now. <url>"',
            'label': 1}},             
            {{'text': 'Welcome British Refugees #brexit #EUref <url>',
            'label': 0}},
            {{'text': 'Educate yourself. IS hate anyone who doesn't share their beliefs. Leaving the EU won't stop terrorist attacks <url>',
            'label': 0}},       
            {{'text': 'RT <user>: #IslamIsTheProblem. Islam is a disgrace! This is not a religion it's a cult! #banMuslims  <url>',
            'label': 1}},       
            {{'text': '"After #Brexit #UK can deport #radical #Islam #muslim ""#clerics"" (AKA #isis recruiters #terrorists) #EU ""#humanright "" except for natives"',
            'label': 1}},        
            {{'text': 'F*** Obama and nobody listen to anymore f*** that Muslim <url>',
            'label': 1}},        
            {{'text': '"#Brexit Facts about Pakistanis 1) Everyone hates them 2) Pakis only hate Hindus 3) Pakis dogs of Arabians"',
            'label': 1}},
            {{'text': '"#Brexit will impact so many different things stock market, trade, employment, immigration etc Thursday is a big day <url>"',
            'label': 0}},       
            {{'text': 'No more Islam #HENK <url>',
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
df.to_csv('/scratch/p281734/brexit_llama3_ambiguous_random.csv', index=False)


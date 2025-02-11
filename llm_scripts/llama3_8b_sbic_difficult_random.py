from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch
import pandas as pd

model = "meta-llama/Meta-Llama-3-8B-Instruct"
#model = "~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(model)

df = pd.read_csv('/scratch/p281734/SBIC_binary_aggr_test.csv', sep=',', header=0)
#df = pd.read_csv('/scratch/p281734/prova.csv', sep=',', header=0)

responses = []

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

for index, row in df.iterrows():
    text = row['post']
    messages = [
        {"role": "system", "content": "You are an AI expert in text classification and content moderation."},
        {"role": "user", "content": """
        You will read a text. Is the text offensive?
        Answer only with 0 for NOT and 1 for YES.
        Do not write an introduction or summary. Classify always the text.
        Here are twenty examples:
        {{ 'response': [
            {{'text': '"Fact: believe it or not most of you will not be incel for long Most of you will soon get laid, not necessarily become normie or chad but at least escape being incel"',
            'label': 1}},
            {{'text': '"If Putin picked out a reality TV star and colluded with him to take the American presidency with the entire federal spy apparatus watching the whole time and yet left no evidence whatsoever,  maybe we should just congratulate him and let him build that wall along the Mexican border he wanted so much"',
            'label': 0}},   
            {{'text': 'Can't wait for certain people to get the reality check they deserve when they leave Newport and they aren't the hot shit they think they are',
            'label': 0}},                 
            {{'text': '"If it didn't work out, then it didn't work out, stop bein a mad hoe and move on"',
            'label': 1}},
            {{'text': 'Some of you girls think being a bad bitch is all there is in life',
            'label': 0}},
            {{'text': '"Who gives a flying fuck? @Rhace138 Again, call me sexist but men shouldn't get lower back tattoos"',
            'label': 1}},
            {{'text': 'i 've been trying to . <repeat> get my daughter to call me daddy . <repeat> appearrantly getting * * our * * mom pregnant does n't make her * * my * * daughter as well . <repeat> ~*boos*~',
            'label': 1}},
            {{'text': 'The gym has my elbow FUCKED up. #andhaileysabitch #goodbye',
            'label': 1}},                
            {{'text': 'when harmos talk about feminism no doubt that it makes me fucking cringe',
            'label': 0}},                
            {{'text': 'Geeze I am watching judge Judy and she's very racist LOL she's getting more cranky every day',
            'label': 1}},
            {{'text': 'If we aren't going to add new mods we need to go private The brigading is out of fucking control  Sick of seeing anti incel comments with 20+ upvotes.',
            'label': 1}},
            {{'text': 'would do a snake if someone﻿ held the fangs back',
            'label': 0}},
            {{'text': '"Starbucks employee (male):: What can I get you Sir? Me (transitioning female to male):: [order] SE:: Oh I'm sorry Miss. Me:: Uh...it's sir. SE:: Here's your coffee MISS sorry again MISS. My friend thinks I confused him. I'm also a college student so if I wasn't sir, I should have at least have been ma'am."',
            'label': 0}},        
            {{'text': '"How would you call a dog if it has long neck , can break all things and can dive into water ?"',
            'label': 0}},       
            {{'text': 'Friends....if you get a chance please follow @whitezombie a friend who just deleted his FB account...after having Zuckerburger try to me his momma one too many times THANKS',
            'label': 0}},             
            {{'text': 'Do girls just have this part of their brain that makes them think about thinks to annoy them... coz I certainly fucking do&#129300;&#128173;&#128580;',
            'label': 1}},
            {{'text': '"Oh, darling MRAs. You making jabs at my appearance won't keep me from working to provide tools so others can shut you down. &lt;3"',
            'label': 1}},       
            {{'text': '"yahoo.com/news/m/d03f5ed5-08fb-3a72-a24e-b7a9951cf8f8/ss_artillery-strikes-against.html … Coalition left not a stone in Raqqa bombing them to the ground, like they bombed Dresden in 1945, already defeated.#Brutality #Civilians"',
            'label': 1}},       
            {{'text': 'why did hitler hate oranges ? because it had juice(jews ) in it .',
            'label': 0}},        
            {{'text': 'Spanish time?',
            'label': 0}}        
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

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
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
df.to_csv('/scratch/p281734/sbic_llama3_difficult_random.csv', index=False)


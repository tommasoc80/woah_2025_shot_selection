from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch
import pandas as pd

model = "meta-llama/Llama-3.2-3B-Instruct"
#model = "~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(model)

df = pd.read_csv('/scratch/p281734/edos_aggregated_test.csv', sep=',', header=0)
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
        You will read a text. Is the text sexist?.
        Answer only with 0 for NOT and 1 for YES.
        Do not write an introduction or summary. Classify always the text.
        Here are twenty examples:
        {{ 'response': [
            {{'text': '"All of those SJW purple haired foids over IT have 5'2"" balding curry janitor bfs and that's a fact."',
            'label': 1}},
            {{'text': 'Bet she'd be hot if she lost all the fat.',
            'label': 1}},        
            {{'text': 'You need to do five dried grams of shrooms and smoke as much Afghani has as your lungs can handle. Beat the Jews on the ballot box dude. Not sure it this helps :)',
            'label': 0}},
            {{'text': '"The difference is that there is *never* a reason to use force to “discipline” your partner. They’re a grown adult, not a child"',
            'label': 0}},
            {{'text': 'Janis #Joplin: the Psychedelic Tranny [URL] #Transgenderagenda #Transgender #transpocalypse',
            'label': 1}},
            {{'text': '"I’ve stopped texting women all together, I just call them now. Saves me the headache. Edit: grammar"',
            'label': 0}},                
            {{'text': '"there are men in ME and Africa who get killed in wars. If you care about men rights, why don't you fight for their rights instead of sitting on your first world ass and bitching about how that on woman was too rude to you hence you hate all women"',
            'label': 1}},
            {{'text': 'The Donald calls out Billary as a lying sack of fetid feces. [URL]',
            'label': 0}},
            {{'text': '12 years is an absolute joke - even hanging is too good for this evil p.o.s.',
            'label': 0}},
            {{'text': '"These fuck sponges really believe this shit don't they? To them women are just holes that fuck everyone, and yet somehow they can't get laid. What happened to these fuckers to make them this twisted, to make them hate women with such a passion."',
            'label': 0}},        
            {{'text': '"This is another reason why women seem to get over you quicker. They already finished the relationship 6 months ago and were just preparing the escape pod while still smiling, laughing and fucking you."',
            'label': 1}},       
            {{'text': 'I suggest you stop. This redditor is always spouting anti-women rhetoric. He always gets downvoted. He always bitches. Leave him to stew in his own hatred.',
            'label': 1}},             
            {{'text': '"Except for procreation, women aren't necessarily for the society to be functional... a year without men will return it to stone age."',
            'label': 1}},
            {{'text': '"if you burn in fire, a women will just walk away and laugh about it."',
            'label': 1}},       
            {{'text': 'If there's ANY amnesty for DACA mud people....I can guarantee you that I'm going to be anti-Trump. He CANNOT cuck on immigration or the Wall.',
            'label': 0}},       
            {{'text': 'Okay retard Andy over here going into subreddits to comment that they dont like something. Yet I'm a joke. Okay bud. Whatever you say.',
            'label': 0}},        
            {{'text': 'she gets medical cannibis cause her pussy smells like a dead goat sanz thats why i tell the lot of yous [URL]',
            'label': 1}},        
            {{'text': '"[URL] Hu'tegh SoHvaD pagh maQam! 'oH wIghaj, liberal toDSaH! vaj fuck! Aaaargh, we can't stand you anymore! We've had it with you, #liberal idiots! So, fuck off!) #Discovery #StarTrek #Klingon #warriors #racist #Trump #supporters"',
            'label': 0}},
            {{'text': '"A bipolar borderline bitch? Yep, been there, its my deepest scar. Cunts be cunts...expect bad behavior or wise up and go alone and let some other guy get a knot in their stomach at night while trying to sleep wondering where shes at"',
            'label': 1}},       
            {{'text': 'Piss off fakecel.',
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
df.to_csv('/scratch/p281734/edos_llama3_random_random.csv', index=False)


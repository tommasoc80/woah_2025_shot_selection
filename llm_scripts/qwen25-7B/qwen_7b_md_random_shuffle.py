from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import pandas as pd

model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
#    quantization_config=bnb_config,
    torch_dtype="auto",
    device_map="auto"
#    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

df = pd.read_csv('/scratch/p281734/MD_hard_label_test.csv', sep=',', header=0)
#df = pd.read_csv('/scratch/p281734/prova.csv', sep=',', header=0)

responses = []

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

for index, row in df.iterrows():
    text = row['text']
    messages = [
        {"role": "system", "content": "You are an AI expert in text classification and content moderation."},
        {"role": "user", "content": """
        You will read a text about one of these topics: Black Lives Matter, the 2020 US Presidential Elections, the COVID-19 pandemic. Is the text offensive?
        Answer only with 0 for NOT and 1 for YES.
        Do not write an introduction or summary. Classify always the text.
        Here are twenty examples:
        {{ 'response': [
            {{'text': '"<user> <user> That’s good. But so many of Trump’s lackeys and enablers and sycophants sailed right back into Congress, paying no price for their subservience to a wannabe dictator. They won’t be the least bit humbled, and they will continue doing harm."',
            'label': 1}},
            {{'text': '"<user> <user> <user> He has 4000 lawyers. If AZ doesn't call the race soon, something bad is happening."',
            'label': 0}},        
            {{'text': '<user> <user> NO JUSTICE NO PEACE #BlackLivesMatter  #VidasNegrasImportam',
            'label': 0}},
            {{'text': '"<user> <user> Can’t go there, it’s already filled up with the corrupt Democrats! 🤷‍♀️"',
            'label': 1}},
            {{'text': '"<user> Go to an event at 1600 COVID Ave? Hell, I don't want to go to an event anywhere right now."',
            'label': 0}},
            {{'text': 'What the actual f*** <url>',
            'label': 1}},                
            {{'text': '#RonDeSantis is an idiot like the #FakePresident telling counties to hold back covid numbers in Florida WTF',
            'label': 1}},
            {{'text': '"Black people ""trump and republicans are racist, VOTE BIDEN LIKE YOUR LIFE DEPENDS ON IT""                               BIDEN:"',
            'label': 0}},
            {{'text': '"Unless <user> is also going to get rid of the flu this moron is gonna have to shut school down forever. (Imbecile is hurting the poorest least advantaged children the most, but that is his goal) <url>"',
            'label': 1}},
            {{'text': '<user> #sleepingjoe cry no more 😂😂😂 you so awful😂',
            'label': 1}},        
            {{'text': '<user> Nah Joe....you got your ass whooped last night. In one night...Trump proved Obama corrupt and your family to be hot trash.',
            'label': 1}},                    
            {{'text': '<user> You're winning this election Sir',
            'label': 0}},
            {{'text': 'This was a complete train wreck.  <user> was all over the place and Chris Wallace sucked as a moderator.  Chris Wallace was clearly in the tank for Biden.  It was one of the worse debates I’ve ever seen.  #Trump2020',
            'label': 0}},       
            {{'text': 'High off my coronavirus ft Mariah Carey okkk😂😂😂😂😂',
            'label': 0}},       
            {{'text': '"Because NO ONE in the media challenges him on his daily lies. Your colleague, Chris Wallace, was a complete disgrace. Biden refuses to appear on any of Fox prime time shows where he would be held to his lies. Everyone else kisses his as*. <url>"',
            'label': 0}},        
            {{'text': 'The left are evil people',
            'label': 1}},        
            {{'text': '"<user> Disgusting behavior is a cop murdering an innocent black men with his knee in his neck for 9 minutes, that’s what’s disgusting. George Floyd was a minister.   <url> <url>"',
            'label': 0}},
            {{'text': '"<user> <user> I'm disgusted the only campaign that trump has is to claim voter fraud, and toss created dirt about <user> and <user> !! PS: My Harris &amp; Biden action figures are stomping on my voodoo doll of trump!"',
            'label': 1}},       
            {{'text': '<user> <user> Better idea; donate your time and/or money to #BlackLivesMatter Sign an online petition even. That would piss him off a million times more plus it’s actually productive.',
            'label': 0}},
            {{'text': 'The hits just keep coming...what a turd.',
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


    outputs = pipeline(
        prompt,
        max_new_tokens=256,
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
df.to_csv('/scratch/p281734/md_qwen7b_random_random.csv', index=False)


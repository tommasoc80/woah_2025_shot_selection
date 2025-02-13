from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch
import pandas as pd

model = "meta-llama/Meta-Llama-3-8B-Instruct"
#model = "~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(model)

df = pd.read_csv('/scratch/p281734/MD_hard_label_test.csv', sep=',', header=0)
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
        You will read a text about one of these topics: Black Lives Matter, the 2020 US Presidential Elections, the COVID-19 pandemic. Is the text offensive?
        Answer only with 0 for NOT and 1 for YES.
        Do not write an introduction or summary. Classify always the text.
        Here are twenty examples:
        {{ 'response': [
            {{'text': '"Wow he had them fainting from #HeatStroke in #Florida, and freezing with #Exposure in #Omaha and #Pennsylvania! Those he doesn't kill with #ExposureToTheElements he'll kill later with #Covid. This is how much #MAGA cares for his supporters, for Americans. #MAGA #TrumpDespisesYou! <url>"',
            'label': 0}},
            {{'text': 'So proud of my city for staying strong and continuing to protest all of last week and this week. #BlackLivesMattter #fuck12',
            'label': 0}},        
            {{'text': '"#Vote <user> Trump is incompetent, illegitimate, rude and terribly corrupt. #Nevada #debates"',
            'label': 0}},
            {{'text': 'Ion know why that anonymous dude scare me that shit ain’t normal😂',
            'label': 0}},
            {{'text': '"<user> <user> Such a stupid, stupid man. He truly has no understanding of the #Constitution or how #America works. The #GOP is complacent in every single thing he has done since the #Impeachment. 100,000+ dead #Americans from #COVID19 &amp; now another #AbuseOfPower w/Social Media. #GOP"',
            'label': 0}},
            {{'text': 'Of course he’s golfing instead of working on covid mitigation or his baseless post-election legal strategy.   I love how Trump Supporters think he’s gives a fuck about them.   He’s grifting you to the very end.   You’ve been played.',
            'label': 0}},                
            {{'text': '<user> <user> <user> <user> You’re disgraceful',
            'label': 0}},
            {{'text': 'Because religion is *magical*',
            'label': 0}},
            {{'text': 'Hey <user> you'd better off if you hid somewhere &amp; stopped talking. Your people can't cover for you when you making a fool out of yourself.',
            'label': 0}},
            {{'text': '"<user> Ugh whoever this person is -----&gt; u are SO RUDE <user> Speaking up here is allyship, not patronization. Get it straight.   Everyone should be speaking up about this &amp; I wish Gabbie didn't even bother to respond to you, SMH. #BLM  #BlackLivesMatter <url>"',
            'label': 0}},        
            {{'text': '<user> You are a liar!!! #BidenCrimeFamilyExposed  Biden is a threat to national security!! #Trump2020 and #VoteRed2020ToSaveAmerica',
            'label': 1}},       
            {{'text': 'What in the hell is wrong with Joe. Joe Biden who let him out of the basement. He is absolutely nuts to say that about King to even compare him to this thug with a criminal record a mile long King never robbed a pregnant woman 🤰🏿',
            'label': 1}},             
            {{'text': '"For everyone on anonymous account , this is a social experiment, please change your profile name to "" HOMOSEXUAL ""  while u go on replying these insensitive cisHet who are homophobic.  It’s Time to choke them with the visibility of that word. 🏳️‍🌈"',
            'label': 1}},
            {{'text': 'No way these Chinese people are not trying to reduce their population... how do you suddenly report 103 new coronavirus deaths in a day???',
            'label': 1}},       
            {{'text': '"<user> <user> Let’s hope. However, tens of millions of voters still supporting a *leader* who is openly sexist, racist, unable to feel empathy and opposed to the separation of powers, are casting a very dark shadow over this victory."',
            'label': 1}},       
            {{'text': '<user> Your future is behind bars',
            'label': 1}},        
            {{'text': '"<user> You’re a complete failure. You are literally misrepresenting the COVID-19 numbers for the moron in the WH.  As for me and my family, we will keep to the guidelines and not go out to party on the beach. <url>"',
            'label': 1}},        
            {{'text': '"<user> Speak for your own corrupt little black heart. We CHuMPS are thankful that yet another of our LEOs didn't lose their life because of a knife wielding thug. Looks like your ""supporters"" are grieving &amp; stealing thru their tears. How touching. TRUmp2020🇺🇲❤ <url> <url>"',
            'label': 1}},
            {{'text': '"Yes, Fauci is a liar... Boom, Boom. <url>"',
            'label': 1}},       
            {{'text': '"<user> <user> GUILTY! <user> de Blasio KILLED 5,000 New Yorkers by FORCING Nursing Homes to accept #COVID19!  He also KILLED dozens and injured hundreds by inciting Democrat Criminals to violence against the #NYPD!  He facilitated #ANTIFA &amp; #BLM Democrat Terrorists! <url>"',
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
df.to_csv('/scratch/p281734/md_llama3_difficult_not_first.csv', index=False)


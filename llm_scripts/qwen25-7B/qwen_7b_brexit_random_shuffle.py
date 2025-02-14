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

df = pd.read_csv('/scratch/p281734/Brexit_hard_label_test.csv', sep=',', header=0)
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
        You will read a text related to Brexit. Does the text contain hate speech?
        Answer only with 0 for NOT and 1 for YES.
        Do not write an introduction or summary. Classify always the text.
        Here are twenty examples:
        {{ 'response': [
            {{'text': '"#foxnews #oreillyfactor #Brexit this was all about stopping the immigration of Muslims, Radical Islam &amp; sharia law. #Trump2016 #TrumpTrain"',
            'label': 1}},
            {{'text': 'To get the European market involves free movement of people. Can a single #leave voter explain how that will reduce immigration ? #brexit',
            'label': 0}},        
            {{'text': 'You see this woman? Every time they photograph her she's doing the Globalist pyramid sign:Mrs Muslim loving Hitler <url>',
            'label': 1}},
            {{'text': 'Can someone get Farage to deport his German wife and children. #brexit',
            'label': 1}},
            {{'text': '"RT <user>: PAKIS OUT WAIFUS IN #Brexit <url>"',
            'label': 0}},
            {{'text': '"<user> <user>  London still has the muslim mayor. Get rid of him, and we'll come to visit. #Brexit"',
            'label': 1}},                
            {{'text': '"<user> The Brits have their #Brexit, but what will they do with it? Will they finally put a stop to rampant Muslim immigration?"',
            'label': 0}},
            {{'text': 'The implications of #Brexit for refugees and migrants <url>',
            'label': 0}},
            {{'text': 'No one likes a globalist!   Congrats England!  Now get that commie Muslim mayor out of London. #Brexit <url>',
            'label': 1}},
            {{'text': 'We must stop Islam because is trying to get dominate status all over the world as The Quran and Muhammed are ruling. <url>',
            'label': 1}},        
            {{'text': '".<user>: Europeans expect us, politicians, to do our jobs. On unemployment, refugees &amp; terrorism - tasks are clear. #Brexit #UKref"',
            'label': 0}},        
            {{'text': '#iVoted #VoteLeave because the jihadi endorsing terrorist sympathisers are begging us to #Remain. #EUref #Brexit <url>',
            'label': 0}},
            {{'text': 'Where is Lallana? Did we deport him for having a foreign name? #brexit #ENG #ISL',
            'label': 0}},       
            {{'text': '"THANK ALLAH for #Brexit - Noor Salman, Pulse terrorist wife. (Yeah the one who texted husband during Ramadan Rampage, whom AG Lynch Lost!)"',
            'label': 1}},       
            {{'text': '"If calling Radical Islam evil causes MORE Muslims 2 become radicalized, Why has the OPPOSITE been true 4 Christians? <url>"',
            'label': 0}},        
            {{'text': '"3 muslims have attacked and killed dozens at #Istanbul airport, will #VoteRemain and the MSM blame islam for this like they did #Brexit?"',
            'label': 0}},        
            {{'text': 'UK Gives Obama &amp; Islam A Big ‘F**K YOU!’ – Votes To LEAVE The EU &amp; Muslim Terror Refugees – #BREXIT <url>',
            'label': 1}},
            {{'text': 'What estimate do the great &amp; good make of opposition to radical Islam as factor in #Brexit? #ElephantInRoom <url>',
            'label': 1}},       
            {{'text': '"I reject: - racism is about education - #Brexit is about lack of education - differing views on immigration equate to racism"',
            'label': 0}},        
            {{'text': 'F*** Obama and nobody listen to anymore f*** that Muslim <url>',
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
df.to_csv('/scratch/p281734/brexit_qwen7b_random_random.csv', index=False)


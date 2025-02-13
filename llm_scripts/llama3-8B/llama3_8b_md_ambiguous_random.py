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
            {{'text': '"Y’all. My moms great-great nephew is in the NICU (idk what that is to me,3rd cousin? Idk) he’s not even a month fucking old and he’s on o2 with some seriously fucked up lungs. Idk if they have tested for #COVID but I stg STAY HOME PPL THIS IS WHY #Babyboy #ShelterInPlace <url>"',
            'label': 1}},
            {{'text': 'This seems like a pretty thin platform. I predict it will break under the weight of Trump. I wonder How long can they pretend the Pandemic isn't relevant?',
            'label': 1}},        
            {{'text': '<user> <user> AND when Arizona is called for <user> you can be sure that War hero and patriot John McCain will be giving killer Trump a HUGE f*ck you to him from above!!! 😂🤣😂🤣😂',
            'label': 1}},
            {{'text': '"COVID Spotlight: Cherylee Sherry, MCHES®, Minnesota Department of Health  <url> #CHES #COVID #HealthEducation <url>"',
            'label': 0}},
            {{'text': '"<user> <user> <user> Will one of you please appeal somehow to the folks Trump told to ""standby""? Defuse them somehow. 🤷🏻‍♀️"',
            'label': 0}},
            {{'text': '"The left would call them coons and house Negroes, not really black, pawns for the white supremacy, blah blah blah.   They have no creed. The only way for any of the lefts little dog-whistle theories to hold weight is for everything that’s true to be false."',
            'label': 1}},                
            {{'text': '"#BlakeLivesMatter America the beautiful! Fck that flag! Grown a§ ppl worshiping a piece of cloth over human lives is more disgraceful, dishonorable, disrespectful, despicable, than kneeling during a song. A SONG! Mind blowing! #DISMISSED #BlackLivesMatter <url>"',
            'label': 1}},
            {{'text': '4th Industrial REvolution. They pay you in digital currency for doing tasks sent to you on your smartphone in order to make you expend energy which is then used to mine the digital currency they'll pay you with. A 'perfectly balanced' hell on Earth. #Technocracy <url>',
            'label': 0}},
            {{'text': 'Black lives do matter all lives do matter. I just hope this dreadful virus that has no respect for any lives does not take more victims from those who joined mass protests on our streets.Those protests to remind us all how important life is! #BlackLivesMatter #AllLivesMattter',
            'label': 0}},
            {{'text': '<user> <user> He fucking said it was a great day for George Floyd.  It's never a great day to be dead after being brutally attacked.',
            'label': 1}},        
            {{'text': '"<user> The POS just said, paraphrasing, that children don't catch COVID very easily, and don't transmit it.  GET THE F*** OUT OF HERE."',
            'label': 1}},       
            {{'text': '<user> just saw you were looking forward to the Knicks screwing this up',
            'label': 0}},             
            {{'text': '"Minnesota Governor authorizes 'full mobilization' of state's National Guard, says protests no longer about death of George Floyd  <url> ⁦<user> so you’re saying a minority of locals stole giant TVs cases of alcohol, etc? 😂"',
            'label': 0}},
            {{'text': '<user> and <user> are both pedophiles but Americans literally ignore this and think they are voting for lesser evil v. greater evil. It's fucking disgusting.',
            'label': 1}},       
            {{'text': '<user> <user> Don’t fuck with Anonymous.',
            'label': 1}},       
            {{'text': 'When it’s fashionable eh? Fuck off. <url>',
            'label': 1}},        
            {{'text': '<user> Las las George Floyd ancestors fit dey traced to Naija so s guy wan collect funds on dia behalf origin chapter',
            'label': 0}},        
            {{'text': '<user> <user> <user> <user> Decades ago the old guard was in control. There were protests back in the 60'&amp;70's. They tried back then. It took the invention of social media and cell phones to get everyone involved. George Floyd died right in front of the eyes of the world. Didn't bother Trump much to see it.',
            'label': 0}},
            {{'text': '"Fuckbook and Suckerberg are leading the world in disinformation, lies and propaganda. All in the name of surveillance and controlling your every movement."',
            'label': 1}},       
            {{'text': '#BrownLivesMatterToo The movement Black Lives Matter is racist itself by its name It does mean that lives of rest of colored people doesn't matter to Americans But still #BlackLivesMatter',
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
df.to_csv('/scratch/p281734/md_llama3_ambiguous_random.csv', index=False)


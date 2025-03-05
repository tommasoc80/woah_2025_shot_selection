from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import pandas as pd

model_name = "Qwen/Qwen2.5-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
#    quantization_config=bnb_config,
    torch_dtype="auto",
    device_map="auto"
#    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

df = pd.read_csv('/scratch/p281734/SBIC_binary_aggr_test.csv', sep=',', header=0)
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
    text = row['post']
    messages = [
        {"role": "system", "content": "You are an AI expert in text classification and content moderation."},
        {"role": "user", "content": """
        You will read a text. Is the text offensive?
        Answer only with 0 for NOT and 1 for YES.
        Do not write an introduction or summary. 
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
df.to_csv('/scratch/p281734/sbic_qwen3b_zero_shot.csv', index=False)


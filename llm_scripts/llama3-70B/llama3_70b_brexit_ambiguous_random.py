from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd
import argparse

SYSTEM_PROMPT = "You are an AI expert in text classification and content moderation."
INSTRUCTION_PREAMBLE = """You will read a text related to Brexit. Does the text contain hate speech?
Answer only with 0 for NOT and 1 for YES.
Do not write an introduction or summary. Classify always the text.
Here are twenty examples:
```
{'text': '".<user> We'll get exactly the same immigration system. #Brexit told a lie. Gullible fell for it. <user> <user>"', 'label': 0},
{'text': '"Retweeted MUSLIM PATROL (<user>:  #Brexit was not about racism Yeh Right <url> <url>"', 'label': 0},        
{'text': '"RT <user>: Calais Muslim Migrants Block Roads, Destroy Cars, Shout 'Fuck UK' #BanIslam #StopIslam #Brexit #VoteLeave <url>"', 'label': 1},
{'text': 'Y is everyone affraid to send this scum of Islam packing <url>', 'label': 1},
{'text': 'RT <user>: amanpour's a pro- refujihadi rapefugee dhimmi. See how that works? #MakeBritainGreatAgain <url>', 'label': 1},
{'text': 'misogynistic xenophobic white supremacist terrorist encouraging cowards don't want what is right they just want their way. #Brexit', 'label': 0},    
{'text': 'Michael Gove sets up new law to thwart ‘rogue’ EU courts &amp; deport foreign criminals <url> #Brexit <url>', 'label': 1},
{'text': 'Mother F*CK the Council on Foreign Relations and anything they may have to say. 💩😬💩😬💩 #EUref #Brexit', 'label': 0},
{'text': 'Because immigration is mainly a business matter... <user> in <user> #brexit #immigration #BorisJohnson <url>', 'label': 0},
{'text': 'so equal pay for women would have stopped fears of immigration #brexit #zimmer conflation', 'label': 0},        
{'text': 'A significant day to be discussing foreign policy and #Brexit with former Secretary of State Henry Kissinger. <url>', 'label': 0},       
{'text': '"Oh great, that should make Europe's slide into Islam just that much easier. why don't you just shoot yourselves now. <url>"', 'label': 1}, 
{'text': 'Welcome British Refugees #brexit #EUref <url>', 'label': 0},
{'text': 'Educate yourself. IS hate anyone who doesn't share their beliefs. Leaving the EU won't stop terrorist attacks <url>', 'label': 0},       
{'text': 'RT <user>: #IslamIsTheProblem. Islam is a disgrace! This is not a religion it's a cult! #banMuslims  <url>', 'label': 1},       
{'text': '"After #Brexit #UK can deport #radical #Islam #muslim ""#clerics"" (AKA #isis recruiters #terrorists) #EU ""#humanright "" except for natives"', 'label': 1},        
{'text': 'F*** Obama and nobody listen to anymore f*** that Muslim <url>', 'label': 1},        
{'text': '"#Brexit Facts about Pakistanis 1) Everyone hates them 2) Pakis only hate Hindus 3) Pakis dogs of Arabians"', 'label': 1},
{'text': '"#Brexit will impact so many different things stock market, trade, employment, immigration etc Thursday is a big day <url>"', 'label': 0},       
{'text': 'No more Islam #HENK <url>', 'label': 1}   
```
"""

INSTRUCTION_TEMPLATE = """Now classify the following text:'''{text}'''
Respond only with the valid JSON format below:
{{'text': '{text}', 'label': …}}
"""


def conversations_from_example(example_text):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": INSTRUCTION_PREAMBLE
            + INSTRUCTION_TEMPLATE.format(text=example_text),
        },
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-model-len', type=int, default=2000)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    parser.add_argument('--n-gpus', type=int, default=1)
    args = parser.parse_args()

    df = pd.read_csv("/scratch-shared/tcaselli/Brexit_hard_label_test.csv", sep=",", header=0)
    model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    messages_list = [conversations_from_example(text) for text in df["text"]]
    prompt_token_ids = [
        tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        for messages in messages_list
    ]

    sampling_params = SamplingParams(top_k=1, max_tokens=256)

    llm = LLM(
        model=model_id,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.n_gpus,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
    )

    outputs = llm.generate(
        prompt_token_ids=prompt_token_ids, sampling_params=sampling_params
    )

    responses = [] # store  post-processed output models
    for output in outputs:
        response = output.outputs[0].text
        if str(response).startswith("{"):
            # label_extracted = re.search("\'label\': (\w+)", response)
            keyword = "\'label\':"
            before_keyword, keyword, after_keyword = str(response).partition(keyword)
            #        print(after_keyword.replace("}]","").replace("}", ""))
            clean_answer = after_keyword.replace("}]", "").replace("}", "").replace("\"", "").replace("\n", "").replace(
                "]", "")
            responses.append(clean_answer)
        else:
            #        print("Refused")
            responses.append("Refused")

    df['model_answer'] = responses
    df.to_csv('/scratch-shared/tcaselli/brexit_llama3_70_ambiguous_random.csv', index=False)
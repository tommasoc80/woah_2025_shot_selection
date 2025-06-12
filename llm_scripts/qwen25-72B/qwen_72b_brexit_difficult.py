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
{'text': '"100's terror plots stopped in U.S. – mostly #Moslem refugees <url> ★#MyJihad Islam #Brexit Handwara Uma Bharti #tcot BJP"', 'label': 0},
{'text': 'Of course Dems have. Why you think they like illegals so much? <url>', 'label': 0},        
{'text': '"So my mum voted #brexit to deter the ""hoards of Muslim immigrants"". Don't think she gets that Middle Eastern countries aren't in the EU 😪"', 'label': 0},
{'text': '"Ironically, if this Brit is forced to return, it'll mean #Brexit is directly responsible for a Muslim entering the country."', 'label': 0},
{'text': '"If you voted to #Remain, by definition you are a terrorist and ally of ISIS. #LeaveWins #Brexit 🇬🇧🇬🇧🇬🇧"', 'label': 0},
{'text': 'Merkel needs another 70M person nation to cover ~$12B shortfall from #Brexit. She will fully sell EU to Islam and admit Turkey. 70M Muslims.', 'label': 0},    
{'text': '"Who says the EU cannot be decisive when it comes to kicking foreigners out? #Brexit <url>"', 'label': 0},
{'text': '"<url> 4,500 Muslim invaders that YOU pay for, why not leave the EU with the bill? #Brexit"', 'label': 0},
{'text': '"If calling Radical Islam evil causes MORE Muslims 2 become radicalized, Why has the OPPOSITE been true 4 Christians? <url>"', 'label': 0},
{'text': 'London Muslim Mayor Khan announces name of the new Caliphate 'ScotLondStan' also considering switching to Arabic 4 offl lang #Brexit', 'label': 0},        
{'text': 'RT <user>: Islam has no place in Europe #Brexit <url>', 'label': 1},        
{'text': '"Don't wanna b tricked into white genocide and live like commies 2serve so called Jewish evil master race, thank you😊 <url>"', 'label': 1},
{'text': '"RT <user>: Deport with extreme prejudice. Go live in swedistan, Britain doesn't want you #Brexit <url>"', 'label': 1},       
{'text': 'the scotts have a lot of cucks who exist only to suck pakis off; we need to fix this <url>', 'label': 1},       
{'text': 'DavidJo52951945: ScottAJBlake4 or use some of the EU fees for a proper border force &amp; deport every illegal straight back to France #brexit', 'label': 1},        
{'text': '"THE MAJORITY WILL NEVER allow the Mentally Ill Globalists to turn the world into a SJW and Radical Islam ""SAFE SPACE"" #brexit #Trump2016"', 'label': 1},        
{'text': '"I knew all along that the leavers succeed for #Brexit was all about immigration! Hey u, foreigners get out! <url>"', 'label': 1},
{'text': '"U R rite Rooster!Let's put USA's citizens&amp;interests ahead of all others,stop migrant invasion&amp;create prosperityAgain <url>"', 'label': 1},       
{'text': '.<user> London is the Muslim Brotherhood's Head Office..I wonder when they'll be given their walking papers. #Brexit <user>', 'label': 1},        
{'text': '"RT <user>: Congrats Britain! Now take control of ur muslim problem! Roll on Trump Train! #Brexit #TrumpTrain #Trump2016 #MAGA <url>"', 'label': 1} 
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
    model_id = "Qwen/Qwen2.5-72B-Instruct"

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
        #quantization="bitsandbytes",
        #load_format="bitsandbytes",
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
    df.to_csv('/scratch-shared/tcaselli/brexit_qwen_72_difficult_not_first.csv', index=False)
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
{'text': '"3 muslims have attacked and killed dozens at #Istanbul airport, will #VoteRemain and the MSM blame islam for this like they did #Brexit?"', 'label': 0},
{'text': '"If calling Radical Islam evil causes MORE Muslims 2 become radicalized, Why has the OPPOSITE been true 4 Christians? <url>"', 'label': 0},        
{'text': '"I reject: - racism is about education - #Brexit is about lack of education - differing views on immigration equate to racism"', 'label': 0},
{'text': '"<user> The Brits have their #Brexit, but what will they do with it? Will they finally put a stop to rampant Muslim immigration?"', 'label': 0},
{'text': 'Where is Lallana? Did we deport him for having a foreign name? #brexit #ENG #ISL', 'label': 0},
{'text': 'The implications of #Brexit for refugees and migrants <url>', 'label': 0},    
{'text': '".<user>: Europeans expect us, politicians, to do our jobs. On unemployment, refugees &amp; terrorism - tasks are clear. #Brexit #UKref"', 'label': 0},
{'text': '"RT <user>: PAKIS OUT WAIFUS IN #Brexit <url>"', 'label': 0},
{'text': 'To get the European market involves free movement of people. Can a single #leave voter explain how that will reduce immigration ? #brexit', 'label': 0},
{'text': '#iVoted #VoteLeave because the jihadi endorsing terrorist sympathisers are begging us to #Remain. #EUref #Brexit <url>', 'label': 0},        
{'text': 'What estimate do the great &amp; good make of opposition to radical Islam as factor in #Brexit? #ElephantInRoom <url>', 'label': 1},        
{'text': 'We must stop Islam because is trying to get dominate status all over the world as The Quran and Muhammed are ruling. <url>', 'label': 1},
{'text': '"#foxnews #oreillyfactor #Brexit this was all about stopping the immigration of Muslims, Radical Islam &amp; sharia law. #Trump2016 #TrumpTrain"', 'label': 1},       
{'text': 'F*** Obama and nobody listen to anymore f*** that Muslim <url>', 'label': 1},       
{'text': '"<user> <user>  London still has the muslim mayor. Get rid of him, and we'll come to visit. #Brexit"', 'label': 1},        
{'text': 'Can someone get Farage to deport his German wife and children. #brexit', 'label': 1},        
{'text': 'You see this woman? Every time they photograph her she's doing the Globalist pyramid sign:Mrs Muslim loving Hitler <url>', 'label': 1},
{'text': 'No one likes a globalist!   Congrats England!  Now get that commie Muslim mayor out of London. #Brexit <url>', 'label': 1},       
{'text': '"THANK ALLAH for #Brexit - Noor Salman, Pulse terrorist wife. (Yeah the one who texted husband during Ramadan Rampage, whom AG Lynch Lost!)"', 'label': 1},        
{'text': 'UK Gives Obama &amp; Islam A Big ‘F**K YOU!’ – Votes To LEAVE The EU &amp; Muslim Terror Refugees – #BREXIT <url>', 'label': 1}
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
    df.to_csv('/scratch-shared/tcaselli/brexit_qwen_72_random_not_first.csv', index=False)
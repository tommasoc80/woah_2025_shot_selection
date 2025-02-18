from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd
import argparse

SYSTEM_PROMPT = "You are an AI expert in text classification and content moderation."
INSTRUCTION_PREAMBLE = """You will read a text. Is the text sexist?.
Answer only with 0 for NOT and 1 for YES.
Do not write an introduction or summary. Classify always the text.
Here are twenty examples:
```
{'text': '"the democrat party's planned parenthood makes too much money selling baby parts for them to give it up plus, they can kill off more of their black/brown properties on the taxpayers' dime"', 'label': 0},
{'text': '"This is so fucking cringe, did she really think she was being badass? She sounds like a lunatic"', 'label': 0},        
{'text': '"Male dolphin wakes up next to female whale. ""Aww man! I thought you were just a fat chick! You're a whale?!"" (My apologies to women who are a little heavier than average, I adore you.) ( no really I f-ing ADORE you.)"', 'label': 1},
{'text': 'Can't wait to see if the Ms Marvel doll has her clit chopped off in true Muslim fashion. [URL]', 'label': 1},
{'text': 'Because in the 1940's & 50's it was still OK to beat your wife.', 'label': 1},
{'text': 'Some day hopefully not too long from now this will be a typical question on a high school history test: What percentage Caucasian did Elizabeth Warren's DNA test suggest she was?', 'label': 0},    
{'text': 'You mean all twenty of you chasing the same two women makes you an ""alpha""? Please, you're just a faggot wanting to cross swords with other men in the same hole."', 'label': 0},
{'text': 'I hope Crowder puts the final nail her nigger lovin coffin', 'label': 1},
{'text': '"And men can stop blaming women for their sexual frustrations, women don't give a single shit about them, dumbasses. 😂😂"', 'label': 1},
{'text': 'You almost never see Mediterranean White women fooling around with niggers. The same goes for most (but unfortunately not all) Slavic women.', 'label': 1},        
{'text': 'most Americans that pursue prostitution are privy to Hong Kong just as Europeans would be privy to Red Light District. You're just an ignorant fuck lmao cry more', 'label': 0},        
{'text': '"Yeah, I guess she is supposed to be Obama's Hollywood sex slave."', 'label': 0},
{'text': 'She got her equal rights... And lefts 😂', 'label': 1},       
{'text': '"If he's not already molesting this little girl, I'm sure he will be in the future. I hope this woman comes to her senses and the court system backs her up and she gets complete custody with zero contact."', 'label': 0},       
{'text': 'tranny teaching [USER] [USER] [USER] [USER] [USER]', 'label': 1},        
{'text': '“Waaa girls hate me for my looks no i would not fuck an asian girl asians are gross”"', 'label': 1},        
{'text': '"I can't see any way to do that without it being very ugly, but there is no doubt that women's suffrage has been disastrous. Democracy is a failed experiment, anyway."', 'label': 1},
{'text': 'I spent the first several years after being married trying to live up to crap like this list only to find much later that my wife just wanted me to be a man and take charge of things.', 'label': 0},       
{'text': '"Watch "" #JKRowling is a Vile Piece of Sh*t"" -she lies on #Trump & ... [URL] #GabFam #News Via #PaulWatson [USER]"', 'label': 0},        
{'text': 'fuckin nigger witch', 'label': 1} 
```
"""

INSTRUCTION_TEMPLATE = """Now classify whether the following text is sexist:'''{text}'''
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

    df = pd.read_csv("/scratch/p281734/edos_aggregated_test.csv", sep=",", header=0)
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
    df.to_csv('/scratch/p281734/edos_llama3_70_difficult_random.csv', index=False)
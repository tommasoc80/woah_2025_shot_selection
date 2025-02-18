from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd
import argparse

SYSTEM_PROMPT = "You are an AI expert in text classification and content moderation."
INSTRUCTION_PREAMBLE = """You will read a text. Is the text offensive?
Answer only with 0 for NOT and 1 for YES.
Do not write an introduction or summary. Classify always the text.
Here are twenty examples:
```
{'text': '"I played the Sandy Caps mini game in Paradise Island 2, and my score was: 197 #GameInsight #ParadiseIsland2"', 'label': 0},
{'text': 'Please @netflix @NetflixNL help us to keep the dream alive and give #BatB Beauty and the Beast a new home! #NetflixSaveBatB', 'label': 0},   
{'text': 'That moment when you get English grammar wrong when a student asks... &#128530; EVERYTHING IS MADE UP AND NOTHING MATTERS', 'label': 0},                 
{'text': '"Me: My throat kind of hurt earlier, I think I'm getting sick. Marc: NO! Me: Maybe it was just a glitch.."', 'label': 0},
{'text': '"Made a positive change in my life by not sleeping next to my mobile. I kept it well out of my reach last night, and saved a lot of time."', 'label': 0},
{'text': 'At the beginning of the movie they show that huge group mating scene where they line up the men and women .', 'label': 0},
{'text': 'I have an awesome tattooist in the burbs of Chicago ...', 'label': 0},
{'text': '"I swear, SahBabii's Pull Up Wit Ah Stick instrumental is one of the most relaxing Hip Hop beats I've ever heard."', 'label': 0},                
{'text': 'In which r/drama thinks the hooker abuse story is bullshit', 'label': 0},                
{'text': 'I saw the movie Valkyrie with Tom Cruise in the lead role .', 'label': 0},
{'text': '"You ever stop to think that maybe, just maybe, you're part of the problem? u/N8TheGr8"', 'label': 1},
{'text': 'The guy's not a scientist. And I will bet $.47 that a majority of the other reputable scientists deniers either have no degree whatsoever or a PhD in Biblical Denial from University of Conservative Mormonism in Utah.', 'label': 1},
{'text': 'White People stop giving your time and money to those that hate you.', 'label': 1},        
{'text': 'Hey i just met jew And this is crazy But heres my gas chamber And ill gas your baby', 'label': 1},       
{'text': 'Shit where the gay bitches at ? Come be witcha girl tonight &#128527;&#128069;&#128166;&#128523;', 'label': 1},             
{'text': '"Why are the orphans so worried about hurricane Florence? Like, what's the worst that could happen? They become homeless?"', 'label': 1},
{'text': '"Girl:: Oh hey, so you're bisexual? Me:: Yeah. Girl:: Do you have a crush on me then? Me:: Oh hey, so you're straight? Girl:: Yeah. Me:: Do you have a crush on every guy you meet? Girl at my old school upon finding out I was bisexual. Made me feel like there was no hope left for humanity."', 'label': 1},       
{'text': '"I don’t like going to subway... Last time I went, I choked on Jared’s foot long."', 'label': 1},       
{'text': '"Female privilege #42 When a nerdish female posts a completely irrelevant sheet of her breast feeding she gets praised, if any of us men do the same thing about our masturbation habits and compare it to big macs, calories etc we would be called creeps, this is modern feminism, my fellow gentlemen."', 'label': 1},        
{'text': 'This bad of dicks is wearing sunglasses at the gym. Either he's Cyclops from X-Men or he's finna get backhanded by the hand of God.', 'label': 1} 
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

    df = pd.read_csv("/scratch/p281734/SBIC_binary_aggr_test.csv", sep=",", header=0)
    model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    messages_list = [conversations_from_example(text) for text in df["post"]]
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
    df.to_csv('/scratch/p281734/sbic_llama3_70_random_not_first.csv', index=False)
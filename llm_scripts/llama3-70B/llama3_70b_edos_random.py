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
{'text': '"I’ve stopped texting women all together, I just call them now. Saves me the headache. Edit: grammar"', 'label': 0},
{'text': '"These fuck sponges really believe this shit don't they? To them women are just holes that fuck everyone, and yet somehow they can't get laid. What happened to these fuckers to make them this twisted, to make them hate women with such a passion."', 'label': 0},        
{'text': 'Piss off fakecel.', 'label': 0},
{'text': '"The difference is that there is *never* a reason to use force to “discipline” your partner. They’re a grown adult, not a child"', 'label': 0},
{'text': '"[URL] Hu'tegh SoHvaD pagh maQam! 'oH wIghaj, liberal toDSaH! vaj fuck! Aaaargh, we can't stand you anymore! We've had it with you, #liberal idiots! So, fuck off!) #Discovery #StarTrek #Klingon #warriors #racist #Trump #supporters"', 'label': 0},
{'text': '12 years is an absolute joke - even hanging is too good for this evil p.o.s.', 'label': 0},    
{'text': 'Okay retard Andy over here going into subreddits to comment that they dont like something. Yet I'm a joke. Okay bud. Whatever you say.', 'label': 0},
{'text': 'If there's ANY amnesty for DACA mud people....I can guarantee you that I'm going to be anti-Trump. He CANNOT cuck on immigration or the Wall.', 'label': 0},
{'text': 'The Donald calls out Billary as a lying sack of fetid feces. [URL]', 'label': 0},
{'text': 'You need to do five dried grams of shrooms and smoke as much Afghani has as your lungs can handle. Beat the Jews on the ballot box dude. Not sure it this helps :)', 'label': 0},        
{'text': '"This is another reason why women seem to get over you quicker. They already finished the relationship 6 months ago and were just preparing the escape pod while still smiling, laughing and fucking you."', 'label': 1},       
{'text': '"if you burn in fire, a women will just walk away and laugh about it."', 'label': 1}, 
{'text': '"Except for procreation, women aren't necessarily for the society to be functional... a year without men will return it to stone age."', 'label': 1},
{'text': 'Janis #Joplin: the Psychedelic Tranny [URL] #Transgenderagenda #Transgender #transpocalypse', 'label': 1},       
{'text': 'I suggest you stop. This redditor is always spouting anti-women rhetoric. He always gets downvoted. He always bitches. Leave him to stew in his own hatred.', 'label': 1},       
{'text': 'Bet she'd be hot if she lost all the fat.', 'label': 1},        
{'text': '"All of those SJW purple haired foids over IT have 5'2"" balding curry janitor bfs and that's a fact."', 'label': 1},        
{'text': '"there are men in ME and Africa who get killed in wars. If you care about men rights, why don't you fight for their rights instead of sitting on your first world ass and bitching about how that on woman was too rude to you hence you hate all women"', 'label': 1},
{'text': 'she gets medical cannibis cause her pussy smells like a dead goat sanz thats why i tell the lot of yous [URL]', 'label': 1},       
{'text': '"A bipolar borderline bitch? Yep, been there, its my deepest scar. Cunts be cunts...expect bad behavior or wise up and go alone and let some other guy get a knot in their stomach at night while trying to sleep wondering where shes at"', 'label': 1}
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
    df.to_csv('/scratch/p281734/edos_llama3_70_random_not_first.csv', index=False)
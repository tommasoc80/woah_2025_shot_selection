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
{'text': 'Shouldn’t he be preparing his last will & testament instead of tweeting stupid shit and further embarrassing himself and eroding his reputation and legacy?', 'label': 0}},
{'text': 'Not in the case of Muslim woman', 'label': 0}
{'text': 'Sub-Saharan Africans (even half-breeds) don't exactly bring up images of high quality manufacturing in my mind. What about the rest of you.', 'label': 0}c
{'text': '"And, her daddy was Chief Spreading Bull"', 'label': 0}
{'text': 'It's a 4chan post it probably 404'd', 'label': 0}
{'text': 'Chads prom night ended as a foursome surrounded by dozens of Stacie’s while incels prom night ends in their jaw slipping through the noose', label': 0}
{'text': 'Ok 5’8 very attractive guy vs 6’5 ugly guy the 5’8 guy would get the girls.', 'label': 0}
{'text': 'YouTube removes videos from conspiracy theorist Alex Jones: Infowars website! #CNN is behind this attack! [URL]', 'label': 0}
{'text': 'we don't know if that is ZUMA. we are not sure about that yet.', 'label': 0}
{'text': '""And access to a single market with lucrative free trade arrangements", the nine-year-old cooed as she sold her kittens to the friendly Tom Ka Gai pushcart vendor on the corner."', 'label': 0}}
{'text': 'Here she is the breeding machine No thoughts for how her kids will work rest play eh Africa the same Muslims the same They All sicken me', 'label': 1}     
{'text': '"Not true. TRP is 100% beneficial to her, provided she is the least bit fuckable. um...no, no it's not."', 'label': 1}
{'text': '"If you read the newspaper one day that some crazy bitch murdered her husband, and you recognize a dumped first date in the mugshot, you will thank me."', 'label': 1}
{'text': 'Fucking disgusting you can see the kid trying to hold on single motherhood is a plague avoid that shit AT ALL COST', 'label': 1}
{'text': 'gym thots are fishing with really wide nets-- tread carefully.', 'label': 1}
{'text': '"1&7. Feminists are the worst. For example, that nasty bitch that made bread with her yeast infection riddled coochie goo and threatened to give to her family that she didnt like at Thanksgiving dinner. Sick demented shit!"', 'label': 1}
{'text': '"If women can think the air conditioning can oppress them, there is no limit to the shit they will make up."', 'label': 1}}
{'text': '"Seriously. All women think about is status and sex, and then crotch spawns after they shit their first one out."}. 'label': 1}
{'text': 'Unattractive women don't get bred out and pass on their defects to their offspring.', 'label': 1}
{'text': '"They see but theyll say some bs like"" i know my worth"". Fds women are so sub par that even tbe 6/10 white women on my tinder mog the entre sub forum to oblivion.", 'label': 1}
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

    df = pd.read_csv("/scratch-shared/tcaselli/edos_aggregated_test.csv", sep=",", header=0)
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
    df.to_csv('/scratch-shared/tcaselli/edos_llama3_70_ambiguous_first.csv', index=False)
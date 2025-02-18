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
{'text': '"Seen a vid of a man who girl was cheating, so he decided to expose her by proposing at her job, then have her side dude walk in..."', 'label': 0},
{'text': 'Maybe now because Miranda said that in her acceptance speech in front of everybody all of the award shows will get there shit together &#129335;&#127997;‍♀️', 'label': 1},   
{'text': 'What's the difference between the Titanic and my raping spree? The women and children survived the Titanic.', 'label': 1},                 
{'text': 'Now Liberals are releasing a legion of feminists who are going to malign political adversaries /w sex charges @realDonaldTrump', 'label': 1},
{'text': 'What is zuckerbergs address this dude is dangerous and he has a lot of years left to go', 'label': 1},
{'text': '"St. John 's College Choir , Cambridge , England : Come , thou long expected Jesus : St John 's College Choir , Cambridge - YouTube St John 's College Choir Cambridge - Wise - Prepare ye the way - YouTube Jesus Christ , the Apple Tree Merry Christmas ( almost ) --last week in Advent !!!!!!"', 'label': 0},
{'text': 'What's the worst part about Poland? Having a child.', 'label': 1},
{'text': '"Guy in my Dist &amp; Log class walks in looks around, sighs &amp; says I can't do this shit today I'm going home then turns around and walks out &#128557;"', 'label': 0},                
{'text': 'Sarah Huckabee Sanders at WH Press Briefing: James Comey Gave False Testimony —Ace', 'label': 0},                
{'text': '"i didnt try it yet but i guess i should check it out ; thankx i hope that link gave whoever needed it some info on whatever she ( i dont remember her name , sorry ) needed later"', 'label': 0},
{'text': '.@CriticalKaiju There is literally nothing more creatively pathetic than taking an existing film and just making a black or female version.', 'label': 1},
{'text': 'I prevented someone from being raped today I just stopped chasing them.', 'label': 1},
{'text': '"the <allcaps> black <allcaps> condom <allcaps> a man walks into a whorehouse looking for a little action and he goes up to the house owner and asks,hey , can i get a piece from one of your fine ladies you 've got here ? ”  sorry sir , ” the owner responds ,  but , we 're all full . ”  aw , please i really need some poon tang ! ” and the owner awnswers ,  well , there is one girl left but when you go meet her you have to wear this black condom . ”  whatever , ” the man answers quickly and races upstairs . a few hours later the man comes down and says ' ' wow , that was great . she did n't even make any noise . but why did i have to wear the black condom ? ” and the owner answers ,  respect for the dead . ”"', 'label': 1},        
{'text': '"Black people are good natured, well behaved members of our society, lock this post if you disagree. side note, good luck to those starting back to school"', 'label': 0},       
{'text': '"Don't know what's going on, but it seems like roommates are moving out maybe? Won't know until later I guess"', 'label': 0},             
{'text': '"Ppl need to see that our beliefs are worth fighting for, or they will CONTINUE to accept GOP framing &amp; worldview. We can't be passive!"', 'label': 0},
{'text': 'What are japanese knives used for? more holes to fuck', 'label': 1},       
{'text': 'NEW PLANET OF THE APES TITLE DROP! Planet of the Apes: South Africa.', 'label': 1},       
{'text': 'I have always wondered if there are incels who don't know about this sub.', 'label': 0},        
{'text': 'I wish Kenny Rogers ran a zoo in Nebraska.️', 'label': 0} 
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
    df.to_csv('/scratch/p281734/sbic_llama3_70_ambiguous_random.csv', index=False)
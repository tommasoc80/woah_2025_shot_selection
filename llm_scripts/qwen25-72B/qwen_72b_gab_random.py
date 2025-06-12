from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd
import argparse

SYSTEM_PROMPT = "You are an AI expert in text classification and content moderation."
INSTRUCTION_PREAMBLE = """You will read a text. Does the text contain hate speech?
Answer only with 0 for NOT and 1 for YES.
Do not write an introduction or summary. Classify always the text.
Here are twenty examples:
```
'text': '"A beard was the sign of a soldier in the Roman Republic and Principate. One was expected to be clean shaven as a civilian, but soldiers on campaign were allowed to grow beards as long as it was kept close-cut, not long like some decadent Greek."', 'label': 0},
{'text': '"I’ve often wondered what it would be like for Globalist groupies once they realized they’re living a lie, but now, I’m seeing that it is we patriotic Americans who have been living in a lie for decades... Our country, branches of gov, our MSM and our schools.. ALL are hijacked."', 'label': 0},        
{'text': '"Virtue Signaling isn't a metaphor, it is an adjective- that in this case modifies the noun ""crap"". Stupid fucking boomer."', 'label': 0},
{'text': '"Sometimes businesses get their customers, or maybe customers just get them. Here are 25 times when businesses got it just right! #funnypics #coolstrange #funnysigns #funnybusinesses #lolsigns  25 Times When Businesses Got It ⋆ Cool Strange http://crwd.fr/2rq8W8S"', 'label': 0},
{'text': 'I'm confused about all the doxing. Why?', 'label': 0},
{'text': 'guys...GUYS!!!! Allen Weiesselberg was GRANTED IMMUNITY in the Michael Cohen case! THE SHIT IS GOING TO HIT THE FAN NOW!!! DRUMPF will finally be impeached!!!  These losers actually believe this shit. They haven't found SHIT in TWO YEARS...but *NOW* they're finally going to bring Trump down!!! /wank', 'label': 0},                
{'text': 'Foreign specialists’ may stage chemical attack in Syria in 2 days to frame Assad – Russian MoD https://www.rt.com/news/436876-idlib-fake-chemical-attack/', 'label': 0},
{'text': 'This is helpful. Thanks brother.', 'label': 0},
{'text': 'If it encourages them to clean their toilets I'm all for it.', 'label': 0},
{'text': 'It seems to me that there is a fundamental difference in choosing the locus of control.  Is it imposed externally by others or is it internally by each individual?  And an associated thought is that those who aren't willing to take responsibility, give it away to others. It's easier for them."', 'label': 0},        
{'text': 'The UK:', 'label': 1},       
{'text': 'Throw their asses off a five story building alive like the Muslim’s do to Queers!', 'label': 1},             
{'text': '"What fucking drama queens, letting women vote was a bad idea"', 'label': 1},
{'text': '"Daily reminder that I hate niggers, jews & moslems but what I hate more than anything else are anti-White, White People."', 'label': 1},       
{'text': '"“I’m mentally ill, having mutilated my own genitals because I cannot accept how God made me. You should absolutely take my opinions on politics seriously.” "', 'label': 1},       
{'text': '"So hating Islam is okay. Hating socialism is okay. Hating communism is okay. But hating Jews is low-IQ knuckle dragging. Then, you go on to strawman the massive over-representation of Jews as a ""few jewish names"" and simply refuse to address the JQ, altogether. You try to tie Jews to ""OT ideology"" and say it isn't dangerous, while ignoring the Talmud and Quabbalhism, which are deadly. As is Sabateean Frankism. The Bible refers to jew sects as the Synagogue of Satan, and your average Jew hates white people and Christianity, as they help flood white nations with those Muzzies you hate so much, as the shooter highlighted in his posts and commonly found on other's gab posts, which you called Islamic propaganda. You jew, bro?"', 'label': 1},        
{'text': '"Is there anything that doesn't offend Muslims? Well, except for mass killings committed in the name of Allah, rape and abuse of women and children, and all the evil deeds perpetrated by Islamic adherents... "', 'label': 1},        
{'text': 'Faggot. https://archive.fo/Kb0YP#selection-457.0-1061.1', 'label': 1},
{'text': '"LOVE! Love for America Love for Americans. It is love Obama, you Muslim dung pile a homosexual such as yourself would not know it could not understand it."', 'label': 1},       
{'text': '"They weren't innocents, they weren't protestors, they were Hamas scum who want to obliterate Jews and Israel. Leftwing MSM needs to realise Hamas would indoctrinate all of them if they lived under its governance. http://blogs.timesofisrael.com/defending-its-very-..."', 'label': 1} 
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

    df = pd.read_csv("/scratch-shared/tcaselli/ghc_test.tsv", sep="\t", header=0)
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
       # quantization="bitsandbytes",
       # load_format="bitsandbytes",
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
    df.to_csv('/scratch-shared/tcaselli/gab_qwen_72_random_not_first.csv', index=False)
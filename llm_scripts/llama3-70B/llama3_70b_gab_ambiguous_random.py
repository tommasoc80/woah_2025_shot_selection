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
{'text': 'No! He got back to DC in the afternoon of the 4th!', 'label': 0},
{'text': '"Watch 'Eagle Forum Speaks Out on Con-Con Con, 2353' on YouTube  https://youtu.be/iEWexTMIwU8"', 'label': 0},   
{'text': 'nobody expects shit from women in this rotten society.', 'label': 1},                 
{'text': '"Angela #Merkel admits there ARE ""no-go"" areas in #Germany that ""nobody dares"" to enter:  http://www.dailymail.co.uk/news/article-5446435/Angela-Merkel-admits-no-areas-Germany.html"', 'label': 0},
{'text': 'Child abuse.', 'label': 1},
{'text': 'https://pjmedia.com/trending/massive-cloud-of-midges-descends-on-cleveland-registering-on-weather-radars/  Sounds like Africa. We don't have many bugs in California. It is one of the perks of living here. Almost no mosquitoes. No gnats or midges. You can take a walk without a bug bite.', 'label': 0},
{'text': 'The Louisiana Purchase and the Civil War were loaded with slaves and Jewish Mayors and Generals on each side.  Grant expelled them all in 1862 for war profiteering.  4 generations later we will get this done and expel all the Jews.  World War Jew and the 80 year aftermath is not going to fly.', 'label': 1},
{'text': 'Let her testify and move on.', 'label': 1},                
{'text': 'Proud of #MikePence for telling the kneelers where to stuff it.    We are so DONE with the #NFL at our house. Beyond done.   #USA  #MAGA  #Pence  #Trump', 'label': 0},                
{'text': 'The destiny of our republic and our freedoms depend on how resolutely WE THE PEOPLE dedicate ourselves to our country's most hallowed principles.', 'label': 0},
{'text': 'they arent migrants. they are invading and stealing', 'label': 1},
{'text': 'Veritas needs to get in touch with @pnehlen for an open world of communication. #ShallNotCensor,', 'label': 0},
{'text': '"You have 111 posts - granted I scrolled through them quickly, but which exactly had nice things to about me or about what I had to say, dear?"', 'label': 0},        
{'text': '"Jealous, bigotry and stupidity is their state of mind. Leave them alone, they be lost. "', 'label': 0},       
{'text': 'This old Jew Witch will die lonely.', 'label': 1},             
{'text': '"When half of California's population is foreign criminals, we should be removing at least one of those Senators.."', 'label': 1},
{'text': '"If you need a nonsensical bullshit snowflake pronoun, you can be absolutely certain that I will have no reason to talk to you or about you. In my world, you don't exist. https://www.campusreform.org/?ID=10583"', 'label': 1},       
{'text': '"You wanna bet Hillary will be hanged by 2020? I’ll even give you until January 1, 2021.   I say she won’t be. You say she will be. Whoever is wrong deletes their account? Deal?"', 'label': 1},       
{'text': 'You can feel it.  Oddly saw it in Alex Jones the other day too...', 'label': 0},        
{'text': 'When I responded with the same but female instead to show them how wrong it is they called me a whore and that I oppressed and raped them I as a girl and a lgbtq+ supporter apologize to y'all for these shits ', 'label': 1} 
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

    df = pd.read_csv("/scratch/p281734/ghc_test.tsv", sep="\t", header=0)
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
    df.to_csv('/scratch/p281734/gab_llama3_70_ambiguous_random.csv', index=False)
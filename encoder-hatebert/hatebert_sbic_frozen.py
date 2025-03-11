import pandas as pd
import emoji
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset, DatasetDict
import numpy as np, evaluate
from sklearn.metrics import classification_report



#model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained("GroNLP/hateBERT", num_labels=2)


def clean_samples(data):

    new_samples = []
    content = list(data['post'].values)
    for tweet_message in content:
        tweet_message = re.sub(r'(@\w+)','MENTION', tweet_message)
        tweet_message = re.sub(r'(https\S+)','URL', tweet_message)
        tweet_message = re.sub(r'[0-9]+', 'NUMBER', tweet_message)
        tweet_message = emoji.demojize(tweet_message)
        tweet_message = re.sub(r'#', '', tweet_message)
        tweet_message = re.sub(r'[(#.,\/?!@$%^&*)]', '', tweet_message)

        new_samples.append(tweet_message)

    data["cleaned_text"] = new_samples

    return data

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

def compute_metrics(eval_pred):

    f1 = evaluate.load("f1")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return f1.compute(predictions=predictions, references=labels)

def train_model(tokenized_dataset, data_collator):

    for param in model.bert.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        output_dir="/scratch/p281734/sbic_frozen",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.001,
        do_eval = True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["dev"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(trainer.evaluate())
    predictions, gold, metrics = trainer.predict(test_dataset=tokenized_dataset["test"])
    y_pred = np.argmax(predictions, axis=1)

    return y_pred

if __name__ == '__main__':

    # manually curate split
    train = pd.read_csv('/scratch/p281734/SBIC_binary_aggr_train.csv', delimiter=',', header=0)
    dev = pd.read_csv('/scratch/p281734/SBIC_binary_aggr_train.csv', delimiter=',', header=0)
    test = pd.read_csv('/scratch/p281734/SBIC_binary_aggr_train.csv', delimiter=',', header=0)

    train_clean = clean_samples(train)
    train_clean.drop(columns="post", inplace=True)
    train_clean.rename(columns={"offensiveYN": "label", "cleaned_text": "text"}, inplace=True)

    dev_clean = clean_samples(dev)
    dev_clean.drop(columns="post", inplace=True)
    dev_clean.rename(columns={"offensiveYN": "label", "cleaned_text": "text"}, inplace=True)

    test_clean = clean_samples(test)
    test_clean.drop(columns="post", inplace=True)
    test_clean.rename(columns={"offensiveYN": "label", "cleaned_text": "text"}, inplace=True)

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_clean),
        "dev": Dataset.from_pandas(dev_clean),
        "test": Dataset.from_pandas(test_clean)
        })

    """Tokenization"""

    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")
    tokenized_brexit = dataset.map(preprocess_function, batched=True)

    #print(tokenized_brexit)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    predictions = train_model(tokenized_brexit, data_collator)

    print("Evaluation SBIC Frozen - Test data:")
    print()
    print(classification_report(test_clean["label"], predictions, digits=4))

    test_clean['predictions'] = predictions
    test_clean.to_csv('/scratch/p281734/sbic_hb_frozen.csv', index=False)

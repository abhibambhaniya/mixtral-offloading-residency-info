import argparse
import os
import torch
import numpy as np
import pandas as pd
from categories import subcategories, categories
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import time

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(ntrain, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        label = test_df.iloc[i, test_df.shape[1] - 1]

        # decoder_input_ids = tokenizer("", return_tensors="pt").input_ids.cuda()
        # decoder_input_ids = model._shift_right(decoder_input_ids)
        
        seq_len = input_ids.size(1)
        attention_mask = torch.ones([1, seq_len], dtype=torch.int, device=model.device)

        # model.generate(
        # input_ids=input_ids,
        # attention_mask=attention_mask,
        # past_key_values=past_key_values,
        # streamer=streamer,
        # do_sample=True,
        # temperature=0.9,
        # top_p=0.9,
        # min_new_tokens=1000,
        # max_new_tokens=1000,
        # pad_token_id=tokenizer.eos_token_id,
        # return_dict_in_generate=True,
        # # output_hidden_states=True,
        # # decoder_router_logits=True, 
        # output_router_logits=True,
        # )

        logits = model(
            input_ids=input_ids, attention_mask=attention_mask,
        ).logits.flatten()

        # print(logits)

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[0]],
                        logits[tokenizer("B").input_ids[0]],
                        logits[tokenizer("C").input_ids[0]],
                        logits[tokenizer("D").input_ids[0]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def test_mmlu(model_name, model_loaded, tokenizer):
    ntrain =  5
    data_dir = '/nethome/abambhaniya3/synergy3/Google-MoE/mmlu'
    save_dir = 'results'

    model = model_loaded
    tokenizer = tokenizer
    model.eval()
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, "results_{}".format(model_name))):
        os.makedirs(os.path.join(save_dir, "results_{}".format(model_name)))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None
        )[: ntrain]
        test_df = pd.read_csv(
            os.path.join(data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval(ntrain, subject, model, tokenizer, dev_df, test_df)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(model_name)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(model_name, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                save_dir, "results_{}".format(model_name), "{}.csv".format(subject)
            ),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))

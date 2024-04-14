import argparse
import os
import torch
import numpy as np
import pandas as pd
from categories import subcategories, categories
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import time
from transformers import TextStreamer


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
    prompt = "The following are multiple choice questions (with answers) about {}. Always answer among [A, B, C, D]\n\n".format(
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
    all_times = []
    exp_load_saved = []

    answers = choices[: test_df.shape[1] - 2]
    # print(answers)
    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        # print("TRAIN")
        # print(train_prompt)
        # print("END")
        # print(prompt_end)
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
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
        # print(input_ids.size(), attention_mask.size())
        start_time = time.time()
        outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # streamer=streamer,
        do_sample=False,
        # temperature=0.9,
        # top_p=0.9,
        min_new_tokens=1,
        max_new_tokens=2,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        # output_hidden_states=True,
        # decoder_router_logits=True, 
        output_router_logits=True,
        # output_logits = False
        )
        times = time.time() - start_time
        # print(outputs.sequences[-1][-2])
        # break
        # logits = model(
        #     input_ids=input_ids, attention_mask=attention_mask, 
        # ).logits.flatten()
        total_experts_saved = 0
        for i in outputs['router_logits'][-32:]:
            if len(i) > 1:
                total_experts_saved += i[1]
            else:
                pass
        
        
        pred = tokenizer.decode(outputs.sequences[-1][-2])
        # print(tokenizer.decode(outputs.sequences[-1][-2]))
        # print(tokenizer("A").input_ids[1], tokenizer("B").input_ids[1], tokenizer("C").input_ids[1], tokenizer("D").input_ids[1])

        cor = pred == label
        cors.append(cor)
        all_times.append(times)
        exp_load_saved.append(total_experts_saved)
        # break

    acc = np.mean(cors)
    cors = np.array(cors)

    all_times = np.array(all_times)
    exp_load_saved = np.array(exp_load_saved)

    print("Average accuracy {:.3f} , Average Time:{:0.3f} sec, avg expert load reduced: {}, - {}".format(acc, np.mean(all_times), np.mean(exp_load_saved),  subject))

    return cors, acc, all_times, exp_load_saved


def test_mmlu(model_name, model_loaded, tokenizer, data_dir='/nethome/abambhaniya3/synergy3/Google-MoE/mmlu', save_dir = 'results'):
    ntrain =  5         ## 5 Shot
    

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
        print(f'Starting {subject}, dev size:{dev_df.shape}, Test size:{test_df.shape}')
        
        cors, acc, times, exp_load_saved = eval(ntrain, subject, model, tokenizer, dev_df, test_df)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(model_name)] = cors
        test_df["{}_times".format(model_name)] = times
        test_df["{}_exp_load_red".format(model_name)] = exp_load_saved
        # print(times)
        # for j in range(times.shape[0]):
        #     choice = choices[j]
        #     test_df["{}_choice{}_times".format(model_name, choice)] = times[:, j]
    
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

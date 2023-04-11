import argparse
import os
import numpy as np
import pandas as pd
from categories import subcategories, categories
import time
import urllib
from tqdm.auto import tqdm
import requests
from requests.exceptions import Timeout, ConnectionError
import tyro
import openai
from typing import List

choices = ["A", "B", "C", "D"]

def get_answer(prompt: str) -> str:
    response_obj = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant."}, 
            {"role": "user", "content": prompt}, 
        ], 
        temperature=0.0, 
    )
    response = response_obj["choices"][0]['message']['content'].strip().upper()
    if response not in choices:
        print("Invalid response: {}".format(response))
    return response

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


def eval(
    k_shot, 
    subject, 
    dev_df, 
    test_df, 
    prompt_prefix='', 
    prompt_suffix='', 
):
    cors = []
    all_preds = []

    for i in tqdm(range(test_df.shape[0])):
        # get prompt and make sure it fits
        k = k_shot
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        prompt = prompt_prefix + prompt + prompt_suffix

        label = test_df.iloc[i, test_df.shape[1] - 1]

        pred = get_answer(prompt)

        cor = pred == label
        cors.append(cor)
        all_preds.append(all_preds)

    acc = np.mean(cors)
    cors = np.array(cors)

    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_preds


def main(
    name: str, 
    k_shot: int, 
    data_dir: str, 
    save_dir: str, 
    prompt_prefix: str='', 
    prompt_suffix: str='', 
):

    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, "results_{}".format(name))):
        os.makedirs(os.path.join(save_dir, "results_{}".format(name)))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        print('running:', subject)
        dev_df = pd.read_csv(
            os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None
        )[: k_shot]
        test_df = pd.read_csv(
            os.path.join(data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, preds = eval(
            k_shot, 
            subject, 
            dev_df, 
            test_df, 
            prompt_prefix=prompt_prefix, 
            prompt_suffix=prompt_suffix, 
        )
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(name)] = cors
        test_df["{}_predictions".format(name)] = preds
        test_df.to_csv(
            os.path.join(
                save_dir, "results_{}".format(name), "{}.csv".format(subject)
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


if __name__ == "__main__":
    tyro.cli(main)

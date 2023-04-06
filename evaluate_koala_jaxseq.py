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
from collections import defaultdict

def get_loglikelihood(
    host, 
    n_retries, 
    inputs, 
    max_input_length, 
    max_output_length, 
):
    prefix, text = zip(*inputs)
    prefix = list(prefix)
    text = list(text)
    for _ in range(n_retries):
        response = requests.post(
            urllib.parse.urljoin(host, 'log_probs'), 
            json={
                'in_strs': prefix, 
                'out_strs': text, 
                'max_input_length': max_input_length, 
                'max_output_length': max_output_length, 
            }, 
        ).json()
        if response['status'] == 'success':
            return response['data']
    raise Exception('Failed to get logprobs after {} retries'.format(n_retries))


choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


# def format_example(df, idx, include_answer=True):
#     prompt = df.iloc[idx, 0]
#     k = df.shape[1] - 2
#     for j in range(k):
#         prompt += " USER: {}. {}".format(choices[j], df.iloc[idx, j + 1])
#     prompt += "\nAnswer: GPT:"
#     if include_answer:
#         prompt += " {} </s>".format(df.iloc[idx, k + 1])
#     return prompt


# def gen_prompt(train_df, subject, k=-1):
#     prompt = "BEGINNING OF CONVERSATION: USER: I will give you a series of multiple choice questions about {}. Select the correct answer by responding with one of the four possible answer options (A, B, C, or D). GPT: Ok got. I'm ready! </s>".format(
#         format_subject(subject)
#     )
#     if k == -1:
#         k = train_df.shape[0]
#     for i in range(k):
#         prompt += format_example(train_df, i)
#     return prompt

def format_example(df, idx, include_answer=True):
    prompt = f"Question: {df.iloc[idx, 0]}\n\nChoices:"
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n({}) {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\n\nAnswer:"
    if include_answer:
        prompt += " ({})\n\n".format(df.iloc[idx, k + 1])
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
    host, 
    n_retries, 
    k_shot, 
    subject, 
    dev_df, 
    test_df, 
    prompt_prefix='', 
    prompt_suffix='', 
    max_input_length=1024, 
    max_output_length=1024, 
):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]
    answer_distribution = defaultdict(int)

    for i in tqdm(range(test_df.shape[0])):
        # get prompt and make sure it fits
        k = k_shot
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        # prompt = train_prompt + prompt_end.removesuffix("Answer:")
        prompt = train_prompt + prompt_end
        prompt = prompt_prefix + prompt + prompt_suffix

        label = test_df.iloc[i, test_df.shape[1] - 1]

        inputs = [(prompt, '(A)'), (prompt, '(B)'), (prompt, '(C)'), (prompt, '(D)')]
        probs = get_loglikelihood(
            host, 
            n_retries, 
            inputs, 
            max_input_length, 
            max_output_length, 
        )

        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
        answer_distribution[pred] += 1


        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    print("Answer distribution: {}".format(answer_distribution))

    return cors, acc, all_probs


def main(
    name: str, 
    host: str, 
    k_shot: int, 
    data_dir: str, 
    save_dir: str, 
    n_retries: int=3, 
    prompt_prefix: str='', 
    prompt_suffix: str='', 
    max_input_length: int=1024, 
    max_output_length: int=1024, 
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

        cors, acc, probs = eval(
            host, 
            n_retries, 
            k_shot, 
            subject, 
            dev_df, 
            test_df, 
            prompt_prefix=prompt_prefix, 
            prompt_suffix=prompt_suffix, 
            max_input_length=max_input_length, 
            max_output_length=max_output_length, 
        )
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(name)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(name, choice)] = probs[:, j]
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

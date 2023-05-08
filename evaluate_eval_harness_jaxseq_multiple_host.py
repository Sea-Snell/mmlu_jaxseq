import pprint
import urllib
import requests
import tyro
from lm_eval import evaluator, tasks
from lm_eval.base import LM
from lm_eval.tasks.hendrycks_test import SUBJECTS
from typing import List, Callable, Union
from tqdm.auto import tqdm
import threading
from urllib.parse import urljoin

def identity_fn(x: str) -> str:
    return x

def koala_fn(x: str) -> str:
    return f"BEGINNING OF CONVERSATION: USER: The following are multiple choice questions (with answers).\n\n{x.removesuffix('Answer:')} GPT: Answer:"

input_proc_fs = dict(
    identity=identity_fn, 
    koala=koala_fn, 
)

class LMEvalHarnessInterface(LM):
    def __init__(
        self, 
        hosts: List[str], 
        bsize: int, 
        input_process: Callable[[str], str], 
        max_input_length: int=1024, 
        max_output_length: int=1024, 
        n_retries: int=3, 
    ):
        self.hosts = hosts
        self.bsize = bsize
        self.input_process = input_process
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.n_retries = n_retries
        assert self.n_retries > 0, 'n_retries must be positive'
    
    def request(self, path: str, **kwargs):
        thread_results = {}
        def _request(host: str):
            nonlocal thread_results
            thread_results[host] = requests.post(urljoin(host, path), **kwargs)
        threads = [threading.Thread(target=_request, args=(host,)) for host in self.hosts]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return thread_results[self.hosts[0]]

    def loglikelihood(self, inputs):
        in_strs, out_strs = zip(*inputs)
        in_strs = list(map(self.input_process, in_strs))
        out_strs = list(out_strs)
        results = []
        for i in tqdm(range(0, len(in_strs), self.bsize)):
            in_strs_batch = in_strs[i:(i+self.bsize)]
            out_strs_batch = out_strs[i:(i+self.bsize)]

            did_succeed = False
            for _ in range(self.n_retries):
                response = self.request(
                    'log_probs', 
                    json={
                        'in_strs': in_strs_batch, 
                        'out_strs': out_strs_batch, 
                        'max_input_length': self.max_input_length, 
                        'max_output_length': self.max_output_length, 
                    }, 
                ).json()
                if response['status'] == 'success':
                    did_succeed = True
                    break
            if not did_succeed:
                raise Exception('Failed to get logprobs after {} retries'.format(self.n_retries))
            
            is_greedy = [False]*len(response['data']) # is_greedy is not used in the multiple choice evaluation code.
            results.extend(list(zip(response['data'], is_greedy)))
        return results
    
    def greedy_until(self, inputs):
        raise NotImplementedError
    
    def loglikelihood_rolling(self, inputs):
        raise NotImplementedError


def main(
    host: Union[str, List[str]], 
    k_shot: int, 
    input_process: str, 
    bsize: int=8, 
    n_retries: int=3, 
    max_input_length: int=1024, 
    max_output_length: int=1024, 
):
    if isinstance(host, str):
        host = [host]
    model = LMEvalHarnessInterface(host, bsize, input_proc_fs[input_process], max_input_length, max_output_length, n_retries)
    results = evaluator.evaluate(
        model, tasks.get_task_dict(list(map(lambda x: f"hendrycksTest-{x}", SUBJECTS))), False, k_shot, None, 
    )
    pprint.pprint(results)
    print(sum(map(lambda x: x['acc'], results['results'].values())) / len(results['results']))

if __name__ == "__main__":
    tyro.cli(main)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from utils import *
from model import *
from data import *
from metrics import *
from itertools import chain
import pandas as pd
from typing import Callable


def run_layerwise(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefixes: Prefixes,
    tuned_lens: dict,
    logit_lens: bool = False,
) -> dict:
    """
    Generate the layerwise metrics for the |data| and |model|, while running
    |interventions| during its processing.

    Parameters
    ----------
    model: required, AutoModelForCausalLM
        Model to run inference on.
    tokenizer: required, Union[GPT2Tokenizer, AutoTokenizer]
        Tokenizer corresponding to |model|.
    prefixes: required, Prefixes
        The built prefixes.
    tuned_lens: required, dict
        Dictionary with layer probes, layer norm, and unembed.
    logit_lens: optional, bool
        Indicator whether to use logit lens instead of the tuned lens.

    Returns
    ------
    metrics: dict
        The layerwise metrics: label_space_probs, top_num_labels_match,
        top_1_acc, correct_over_incorrect, cal_correct_over_incorrect, and
        cal_permute.
    """
    start = time.time()

    print("*Get true and false prefix labels.*")
    prefixes_true_labels, prefixes_false_labels = (
        torch.Tensor(prefixes.prefixes_true_labels).type(torch.int64),
        torch.Tensor(prefixes.prefixes_false_labels).type(torch.int64),
    )

    print("*Tokenize inputs.*")
    tokenized_inputs = get_tokenized_inputs(tokenizer, prefixes, model.device.index)

    print("*Get token ids of labels and first token ids of labels.*")
    lab_token_ids = [
        tokenizer.convert_tokens_to_ids(token) for token in prefixes.label_tokens
    ]
    lab_first_token_ids = torch.Tensor([tokens[0] for tokens in lab_token_ids]).type(
        torch.int64
    )

    print("*Get positions of label tokens and positions of preceding token labels.*")
    newline_token = tokenizer("\n")["input_ids"][-1]
    all_label_token_ids = [
        token_id for token_ids in lab_token_ids for token_id in token_ids
    ]
    prec_lab_and_lab_indices = get_prec_lab_and_lab_indices(
        tokenized_inputs, all_label_token_ids, newline_token
    )
    prec_label_indices = [[l[0] for l in labs] for labs in prec_lab_and_lab_indices]

    print(
        "*Run inference and get top 1 intermediate logit, top number of labels intermediate logits, probabilities of labels, and normalized probabilities of labels.*"
    )
    data = (tokenized_inputs, prec_label_indices, lab_first_token_ids)
    (
        top_1_logit,
        top_num_labels_logits,
        probs,
        norm_probs,
    ) = get_label_probs_and_top_logits(model, data, tuned_lens, logit_lens)

    print("*Compute top_1_acc metric.*")
    top_1_acc = get_top_1_acc(top_1_logit, prefixes_true_labels, lab_first_token_ids)

    print("*Compute top_num_labels_match metric.*")
    top_num_labels_match = get_top_num_labels_match(
        top_num_labels_logits, lab_first_token_ids
    )

    print("*Compute label_space metric.*")
    label_space_probs = mean_up_low(torch.sum(probs, -1))

    print("*Get probability of correct labels.*")
    correct_label_probs = get_correct_label_probs(probs, prefixes_true_labels)

    print("*Compute correct_over_incorrect metric.*")
    n_labels = norm_probs.shape[-1]
    correct_over_incorrect = (
        correct_label_probs >= torch.max(probs, -1).values
    ).float()
    correct_over_incorrect = mean_up_low(correct_over_incorrect)
    del correct_label_probs

    print("*Get quantile probabilities of labels for calibration.*")
    n_labels = norm_probs.shape[-1]
    quantiles, means = get_thresholds(norm_probs, n_labels)

    print("*Compute cal_correct_over_incorrect metric.*")
    cal_correct_over_incorrect = get_cal_correct_over_incorrect(
        norm_probs, quantiles, prefixes_true_labels
    )

    print("*Compute cal_permute metric.*")
    cal_permute = get_cal_permute(norm_probs, quantiles, prefixes_false_labels)
    del norm_probs
    del probs
    del quantiles
    del means
    torch.cuda.empty_cache()

    metrics = {
        "label_space_probs": label_space_probs,
        "top_num_labels_match": top_num_labels_match,
        "top_1_acc": top_1_acc,
        "correct_over_incorrect": correct_over_incorrect,
        "cal_correct_over_incorrect": cal_correct_over_incorrect,
        "cal_permute": cal_permute,
    }

    end = time.time()
    print(f"Total time to run: {end - start}.")

    return metrics
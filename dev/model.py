from einops import rearrange
from data import Prefixes
from typing import Callable
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import time
import torch


def get_model_and_tokenizer(
    model_name: str,
    device: int = 0,
) -> (AutoModelForCausalLM, AutoTokenizer,):
    """
    Get the model and tokenizer corresponding to |model_name|.

    Parameters
    ----------
    model_name: required, str
        Name of the model.

    Returns
    ------
    model: AutoModelForCausalLM
        Model corresponding to |model_name|.
    tokenizer: AutoTokenizer
        Tokenizer corresponding to |model_name|.
    """
    if model_name == "EleutherAI/gpt-neox-20b" or model_name == "EleutherAI/pythia-12b-deduped":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(f"cuda:{device}")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(f"cuda:{device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def get_tuned_lens(file_path: str, num_layers: int, context_size: int):
    """
    Get the tuned lens (including the layer norm and unembded).

    Parameters
    ----------
    file_path: required, str
        Path to tuned lens data.
    num_layers: required, int
        Number of layers in the model.
    context_size: required, int
        Size of the context.

    Returns
    ------
    tuned_lens: dict
        Dictionary containing layer probes, layer norm, and unembed.
    """
    tuned_lens_dict = torch.load(file_path)
    layer_probe_weights = torch.stack(
        [tuned_lens_dict["input_probe.weight"]]
        + [tuned_lens_dict[f"layer_probes.{i}.weight"] for i in range(num_layers)]
    )
    layer_probe_biases = torch.stack(
        [tuned_lens_dict["input_probe.bias"]]
        + [tuned_lens_dict[f"layer_probes.{i}.bias"] for i in range(num_layers)]
    )
    ln = torch.nn.LayerNorm((context_size,))
    ln.weight.data = tuned_lens_dict["layer_norm.weight"]
    ln.bias.data = tuned_lens_dict["layer_norm.bias"]
    unembded = tuned_lens_dict["unembedding.weight"]

    tuned_lens = {
        "layer_probe_weights": layer_probe_weights.to("cuda"),
        "layer_probe_biases": layer_probe_biases.to("cuda"),
        "ln": ln.to("cuda"),
        "unembded": unembded.to("cuda"),
    }
    return tuned_lens

def get_tokenized_inputs(
    tokenizer: AutoTokenizer, prefixes: Prefixes, max_length: int = 1024, device: int = 0
) -> list:
    """
    Get the tokenized concatination of |prefixes.true_prefixes| and
    |prefixes.false_prefixes|.

    Parameters
    ----------
    tokenizer: required, AutoTokenizer
        Tokenizer corresponding to the model.
    prefixes: required, Prefixes
        Prefixes object which contains the true prefixes and false prefixes.

    Returns
    ------
    tokenized_inputs: list
        The concatenated, tokenized true prefixes and false prefixes.
    """
    inputs = prefixes.true_prefixes + prefixes.false_prefixes
    tokenized_inputs = [
        tokenizer(inp, return_tensors="pt", padding=True, truncation=True).to(
            f"cuda:{device}"
        )
        for inp in inputs
    ]

    return tokenized_inputs

def get_label_probs_and_top_logits(
    model: AutoModelForCausalLM,
    data: tuple,
    tuned_lens: dict,
    logit_lens: bool = False,
) -> torch.Tensor:
    """
    Run inference on the |model|, with either the |tuned_lens| or
    the logit lens, and get the hidden states at the the tokens 
    preceding the labels.

    Parameters
    ----------
    model: required, AutoModelForCausalLM
        Model to run inference on.
    data: required, tuple
        Tokenized inputs, position of tokens preceding the labels, and the
        token ids corresponding to the labels.
    tuned_lens: required, dict
        Dictionary with layer probes, layer norm, and unembed.
    logit_lens: optional, bool
        Indicator whether to use logit lens instead of the tuned lens.
        
    Returns
    ------
    hidden: torch.Tensor
        Hidden states for the true_prefixes and false_prefixes, of each input,
        at each position in context, and at each layer.
    """
    tokenized_inputs, prec_label_indices, lab_first_token_ids = data
    layer_probe_weights, layer_probe_biases, ln, unembed = (
        tuned_lens["layer_probe_weights"],
        tuned_lens["layer_probe_biases"],
        tuned_lens["ln"],
        tuned_lens["unembded"],
    )

    top_1_logit, top_num_labels_logits, probs, norm_probs = [], [], [], []
    with torch.no_grad():
        for j, (t_inp, indices) in enumerate(zip(tokenized_inputs, prec_label_indices)):

            out = model(**t_inp, output_hidden_states=True)
            hidden = torch.stack(out.hidden_states, dim=1)
            interm = hidden[:, :, indices].float()


            if not logit_lens:
                affine = (
                    interm[:, :-1]
                    + (interm[:, :-1]@layer_probe_weights.mT)
                    + layer_probe_biases.unsqueeze(dim=1).unsqueeze(dim=0)
                )
                last = interm[:, -1].unsqueeze(dim=1)
                interm = torch.cat((affine, last), dim=1)

            logits = ln(interm) @ unembed.T
            probs_ = torch.nn.functional.softmax(logits, dim=3)
            probs_ = probs_[:, :, :, lab_first_token_ids]
            norm_probs_ = (probs_ + 1e-14) / (
                probs_.sum(dim=3, keepdim=True) + (1e-14 * probs_.shape[3])
            )
            probs.append(probs_)
            norm_probs.append(norm_probs_)

            top_1_logit_ = logits.topk(1, dim=3).indices
            top_num_labels_logits_ = logits.topk(
                len(lab_first_token_ids), dim=3
            ).indices
            top_1_logit.append(top_1_logit_)
            top_num_labels_logits.append(top_num_labels_logits_)

    top_1_logit, top_num_labels_logits, probs, norm_probs = [
        rearrange(
            torch.cat(l),
            "(n_prefix n_inputs) n_layer n_demos lab_space_size -> "
            "n_prefix n_inputs n_layer n_demos lab_space_size",
            n_prefix=2,
            n_inputs=len(l) // 2,
        )
        for l in [top_1_logit, top_num_labels_logits, probs, norm_probs]
    ]

    return (top_1_logit, top_num_labels_logits, probs, norm_probs)
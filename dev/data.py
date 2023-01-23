import datasets
import random
import json
import pandas as pd

datasets.disable_progress_bar()


def get_dataset(dataset_params: dict) -> list:
    """
    Get the dataset based off |dataset_params|.

    Parameters
    ----------
    dataset_params: required, dict
        Dataset metadata, specifically the name of the dataset, configuration,
        train or test specification, and if the dataset is on huggingface.

    Returns
    ------
    dataset: list
        List of examples for each class in the dataset.
    """
    set_name, config, train_or_test, on_hugging_face = (
        dataset_params["set_name"],
        dataset_params["config"],
        dataset_params["train_or_test"],
        dataset_params["on_hugging_face"],
    )
    if on_hugging_face:
        raw_data_df = datasets.load_dataset(set_name, config)[train_or_test].to_pandas()
    else:
        raw_data_df = pd.read_csv(f"../datasets/data/{set_name}.csv")

    with open("../datasets/data/data_label_keys.json") as fp:
        content_label_keys = json.load(fp)

    content_keys, label_key = content_label_keys[set_name]
    labels = raw_data_df[label_key].unique()
    labels.sort()

    dataset = []
    for l in labels:
        chained_content = [
            raw_data_df[raw_data_df[label_key] == l][content_keys[i]].values
            for i in range(len(content_keys))
        ]
        interleaved_content = list(zip(*chained_content))
        dataset.append(interleaved_content)

    return dataset


def get_prec_lab_and_lab_indices(
    tokenized_inputs: list,
    all_label_token_ids: list,
    newline_token: int,
) -> list:
    """
    Get the position(s) of the label and the position of the token that
    precedes the label, in each input in |tokenized_inputs|. Each demonstration
    is followed by a period and two new lines, which is utilized for this
    search.

    Parameters
    ----------
    tokenized_inputs : required, list
        Each of the inputs tokenized.
    label_token_ids : required, list
        The token ids corresponding to the labels.
    newline_token : required, int
        The token id corresponding to a newline character.

    Returns
    ------
    prec_lab_and_lab_indices : list
        The positions of the labels and preceding label tokens in each input.
    """
    prec_lab_and_lab_indices = []
    for t_inp in tokenized_inputs:
        input_ids, ii_len = t_inp.input_ids[0], len(t_inp.input_ids[0])
        indices = []
        for j in range(len(input_ids)):
            if (
                j == ii_len - 1
                or (
                    input_ids[j] == newline_token
                    and input_ids[j + 1] == newline_token
                    and input_ids[j + 2] != newline_token
                )
            ):
                period_indx = j
                while input_ids[period_indx] not in all_label_token_ids:
                    period_indx -= 1
                label_indx = period_indx
                while input_ids[label_indx] in all_label_token_ids:
                    label_indx -= 1
                indices.append(list(range(label_indx, period_indx + 1)))
        prec_lab_and_lab_indices.append(indices)

    return prec_lab_and_lab_indices


def get_specialized_indices(label_indx_vals: list, context_pos: int) -> tuple:
    """
    Get,
    1) position of all labels before |context_pos| that match with the label at
    |context_pos|,
    2) position of all preceding label tokens at or before |context_pos| that
    match with the label at |context_pos|,
    3) position of all labels before |context_pos|, and
    4) position of all preceding label tokens at or before |context_pos|.

    Parameters
    ----------
    label_indx_vals: required, list
        The positions of the labels and preceding label tokens, coupled with
        the class that they map to.

    Returns
    ------
    _: tuple
        Tuple containing the same label indices, same preceding label indices,
        label indices, and preceding label indices.
    """
    label_indx_vals_ = label_indx_vals[: context_pos + 1]
    lab_indices_same, prec_lab_indices_same, lab_indices, prec_lab_indices = (
        [],
        [],
        [],
        [],
    )
    indx = 0
    for lab_tups in label_indx_vals_:
        if lab_tups[0][1] == label_indx_vals_[-1][-1][1]:
            lab_indices_same += [indx + i + 1 for i in range(len(lab_tups[1:]))]
            prec_lab_indices_same.append(indx)
        lab_indices += [indx + i + 1 for i in range(len(lab_tups[1:]))]
        prec_lab_indices.append(indx)
        indx += len(lab_tups)

    return (
        lab_indices_same[: -(len(label_indx_vals_[-1]) - 1)],
        lab_indices[: -(len(label_indx_vals_[-1]) - 1)],
        prec_lab_indices_same[:-1],
        prec_lab_indices[:-1],
    )


class Prefixes:
    """
    Class to build the prefixes.

    Attributes
    ----------
    true_prefixes: list
        |dataset_params.num_inputs| true prefixes, each containing
        |demo_params.num_demos| demonstrations.
    false_prefixes: list
        |dataset_params.num_inputs| false prefixes, each containing
        |demo_params.num_demos| demonstrations, with |demo_params.true_percent|
        of the demonstrations being true, and permuted incorrect labels if
        |demo_params.permuted|, otherwise random incorrect labels.
    prefixes_true_labels: list
        Correct label corresponding to each position in context and prefix.
    prefixes_false_labels: list
        Labels used for false demonstrations for each position in context and
        prefix.
    """

    def __init__(
        self,
        dataset: list,
        prompt_params: dict,
        demo_params: dict,
        num_inputs: int,
    ):
        """
        Initializes class.

        Parameters
        ----------
        dataset: required, list
            List of examples for each class in the dataset.
        prompt_params: required, dict
            Prompt metadata, specifically the labels, prompt format, and prefix
            narrative.
        demo_params: required, dict
            Demo metadata, specifically the number of demos, percentage of true
            demonstrations in the false prefix, and an indicator for if permuted
            false labels should be used (as opposed to random false labels).
        num_inputs: required, int
            The number of inputs for each prefix type.

        Returns
        ------
        None
        """
        self.true_prefixes = []
        self.false_prefixes = []
        self.prefixes_true_labels = [[]]
        self.prefixes_false_labels = [[]]

        self.__set_prefixes_and_labels(dataset, prompt_params, demo_params, num_inputs)

    def __get_sample_context(
        self,
        dataset: list,
        prompt_params: dict,
        demo_params: dict,
        num_inputs: int,
    ) -> (list, list):
        """
        Get the unbuilt prefixes (i.e. a list of demonstrations for each
        prefix).

        Parameters
        ----------
        dataset: required, list
            List of examples for each class in the dataset.
        prompt_params: required, dict
            Prompt metadata (see above).
        demo_params: required, dict
            Demo metadata (see above).
        num_inputs: required, int
            The number of inputs for each prefix type.

        Returns
        ------
        context: list
            List of demonstrations for each prefix.
        context_labels: list
            Labels corresponding to the demonstrations in |context|.
        """

        num_demos = demo_params["num_demos"]
        n_labels = len(prompt_params["labels"])
        total_len_const = (100 - len(prompt_params["prompt_format"])) * num_demos

        context = []
        context_labels = []
        for i in range(num_demos):
            demos_sample = [""] * num_inputs
            demos_sample_labels = [""] * num_inputs
            shuffle_inputs = list(range(num_inputs))
            random.shuffle(shuffle_inputs)

            for p, j in enumerate(shuffle_inputs):
                lab = p % n_labels
                sample = random.sample(dataset[lab], 1)[0]

                already_in = sample in [context[k][j] for k in range(i)]
                sample_len = sum([len(s) for s in sample])
                curr_len = sum([len(context[k][j][0]) for k in range(i)])
                too_long = sample_len > ((total_len_const - curr_len) / (num_demos - i))

                alpha = 0
                while already_in or too_long:
                    sample = random.sample(dataset[lab], 1)[0]
                    sample_len = sum([len(s) for s in sample])
                    already_in = sample in [context[k][j] for k in range(i)]
                    too_long = sample_len > (
                        ((total_len_const + alpha) - curr_len) / (num_demos - i)
                    )
                    alpha += i + 1

                demos_sample[j] = sample
                demos_sample_labels[j] = lab

            context.append(demos_sample)
            context_labels.append(demos_sample_labels)

        return (context, context_labels)

    def __set_prefixes_and_labels(
        self,
        dataset: list,
        prompt_params: dict,
        demo_params: dict,
        num_inputs: int,
    ) -> None:
        """
        Sets |self.true_prefixes| and |self.false_prefixes| to the built true
        and false prefixes, and |self.prefixes_true_labels| and |self.prefixes_false_labels|
        to the labels in the true and false prefixes at each position in context.

        Parameters
        ----------
        dataset: required, list
            List of examples for each class in the dataset.
        prompt_params: required, dict
            Prompt metadata (see above).
        demo_params: required, dict
            Demo metadata (see above).
        num_inputs: required, int
            The number of inputs for each prefix type.

        Returns
        ------
        None
        """
        prefix_narrative = prompt_params["prefix_narrative"]
        labels = prompt_params["labels"]
        prompt_format = prompt_params["prompt_format"]

        num_demos = demo_params["num_demos"]
        context, context_labels = self.__get_sample_context(
            dataset,
            prompt_params,
            demo_params,
            num_inputs,
        )

        true_prefixes, false_prefixes = [], []
        prefixes_true_labels, prefixes_false_labels = [], []
        for i in range(num_inputs):
            true_demos, false_demos = (
                ([prefix_narrative], [prefix_narrative])
                if prefix_narrative
                else ([], [])
            )
            demos_true_labels, demos_false_labels = [], []
            cycle_start = random.randrange(1, len(labels))
            for j in range(num_demos):
                true_lab = context_labels[j][i]
                incorrect_labs = list(range(0, true_lab)) + list(
                    range(true_lab + 1, len(labels))
                )
                if demo_params["percent_true"] >= random.uniform(0, 1):
                    false_lab = true_lab
                elif demo_params["permuted"]:
                    false_lab = (true_lab + cycle_start) % len(labels)
                elif demo_params["random_incorrect"]:
                    false_lab = random.choice(incorrect_labs)
                elif demo_params["pure_random"]:
                    false_lab = random.choice([true_lab] + incorrect_labs)
                demos_true_labels.append(true_lab)
                demos_false_labels.append(false_lab)
                true_demo = prompt_format.format(*context[j][i], labels[true_lab])
                false_demo = prompt_format.format(*context[j][i], labels[false_lab])
                true_demos.append(true_demo)
                false_demos.append(false_demo)
            true_prefix = "\n\n".join(true_demos) + "\n\n"
            false_prefix = "\n\n".join(false_demos) + "\n\n"
            prefixes_true_labels.append(demos_true_labels)
            prefixes_false_labels.append(demos_false_labels)
            true_prefixes.append(true_prefix)
            false_prefixes.append(false_prefix)

        self.true_prefixes = true_prefixes
        self.false_prefixes = false_prefixes
        self.prefixes_true_labels = prefixes_true_labels
        self.prefixes_false_labels = prefixes_false_labels
        self.label_tokens = prompt_params["tokens"]

        return
import re
import numpy as np
import time
from openai import AzureOpenAI, OpenAI
from groq import Groq


def get_examples_from_df(df, n_examples):

    # collect used examples to exclude from test dataset
    indexes_to_drop = []

    # collect examples with single fault
    examples = {}

    for c in df.columns[1:]:
        if df[c].sum() == 0:
            continue
        examples[c] = []
        i = df.loc[
            (df[c] == 1) & (df[df.columns[1:][df.columns[1:] != c]].sum(axis=1) == 0)
        ].sample(n_examples)
        indexes_to_drop.append(i.index)
        for e in i.values:
            examples[c].append([e[0], "[" + ",".join(map(str, e[1:])) + "]"])

    # collect examples with multiple faults
    examples_multiple = {}

    t = df.loc[df[df.columns[1:]].sum(axis=1) == 2]
    if t.shape[0] != 0:
        idx = t.sample(n_examples).index
        indexes_to_drop.append(idx)
        t = df.loc[idx].values

        for r in t:
            c = "&".join(df.columns[1:][r[1:] == 1])
            if examples_multiple.get(c) is None:
                examples_multiple[c] = []
            examples_multiple[c].append([r[0], "[" + ",".join(map(str, r[1:])) + "]"])

        # add both single and multiple into one place
        examples.update(examples_multiple)

        indexes_to_drop = np.array(indexes_to_drop).flatten()

    return indexes_to_drop, examples


def parse_result(res):
    """Parse the LLM result"""
    # pattern = r"\[.*?,.*?\]"
    pattern = r"\[[0-9,]*?,.*?\]"
    vec = re.findall(pattern, res)[0]
    return vec.replace(" ", "")
    # return f"[{vec}]".replace(" ", "")


def llm_client(endpoint, api_key, azure_endpoint="", api_version=""):
    """
    Return OpenAI, Azure, or Groq client for inference

    Args:
        endpoint (str): "openai" or "azure" or "groq"
        api_key (str): API Key
        azure_endpoint (str): azure endpoint link. Defaults to "".
        api_version (str): azure endpoint api version. Defaults to "".

    Returns:
        client: client for inference
    """

    if endpoint == "azure":
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )

    if endpoint == "openai":
        return OpenAI(api_key=api_key)

    if endpoint == "groq":
        return Groq(api_key=api_key)


def client_invoke(
    client,
    model_name,
    SystemPrompt,
    Sensors,
    examples_txt,
    UserPrompt,
    instance,
    temperature=0.0,
    max_tokens=4096,
    response_format=None,
):
    """Call client for chat completion"""

    # system prompt
    messages = [
        {
            "role": "system",
            "content": SystemPrompt.format(sensors=Sensors, examples=examples_txt),
        }
    ]

    result = {}

    result["idx"] = instance[0]
    result["requirement"] = instance[1].iloc[0]
    result["true_vector"] = "[" + ",".join(map(str, instance[1].iloc[1:])) + "]"

    # add user prompt
    messages.append(
        {"role": "user", "content": UserPrompt.format(req=result["requirement"])}
    )

    # run LLM
    start_time = time.perf_counter()

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format
    )

    response_time = round(time.perf_counter() - start_time, 6)
    result["ai_response"] = response.choices[0].message.content
    result["pred_vector"] = parse_result(result["ai_response"])
    result["response_time"] = response_time

    result["accuracy"] = result["pred_vector"] == result["true_vector"]

    result.update(response.usage.to_dict())

    # logprobs = response.choices[0].logprobs
    # message = response.choices[0].message
    # message = response.choices[0].message
    # content_filter_results = response.choices[0].content_filter_results

    return result, response


def get_batches(df, SAMPLE_TYPE="random", N_REQS=5):
    batches = []

    while df.shape[0] > N_REQS:
        if SAMPLE_TYPE == "random":
            instances = df.sample(N_REQS)
            df.drop(index=instances.index, inplace=True)
        else:
            instances = df.iloc[:N_REQS, :]
            df.drop(index=instances.index, inplace=True)

        batches.append(instances)

    print("Number of Batches:", len(batches))
    print("Number of Instances left:", df.shape[0])

    return batches


def requirement_text_bulk(batches):
    # split and format batches
    for i, batch in enumerate(batches):
        idx = batch.index.to_list()
        req = batch["requirement"].tolist()
        vec = (
            batch.iloc[:, 1:]
            .apply(lambda x: "[" + ",".join(map(str, x)) + "]", axis=1)
            .to_list()
        )
        batches[i] = (idx, req, vec)

    # requirement text
    req_texts = []
    for i, batch in enumerate(batches):
        req_text = ""
        for i, r in enumerate(batch[1]):
            req_text += f"Requirement {i+1}: {r}.\n"
        req_texts.append(req_text)

    return batches, req_texts


def invoke_bulk(
    model_name,
    client,
    batch,
    SystemPrompt,
    Sensors,
    examples_txt,
    UserPromptBulk,
    req_text,
    temperature=0.0
):
    result = {}

    messages = [
        {
            "role": "system",
            "content": SystemPrompt.format(sensors=Sensors, examples=examples_txt),
        }
    ]

    # add user prompt
    messages.append({"role": "user", "content": UserPromptBulk.format(req=req_text)})

    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )
    response_time = round(time.perf_counter() - start_time, 6)

    vectors = [parse_result(v) for v in response.choices[0].message.content.split("\n")]
    accuracy = [True if gt == v else False for gt, v in zip(batch[2], vectors)]

    result["idx"] = batch[0]
    result["requirement"] = batch[1]
    result["true_vector"] = batch[2]
    result["ai_response"] = response.choices[0].message.content
    result["pred_vector"] = vectors
    result["response_time"] = response_time

    result["accuracy"] = accuracy

    result.update(response.usage.to_dict())

    return result

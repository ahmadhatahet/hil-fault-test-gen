import re
import numpy as np
import time
import json
from datetime import datetime as dt
from openai import AzureOpenAI, OpenAI
from groq import Groq
from tqdm.notebook import tqdm


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
            examples[c].append([e[0], f"[{c}]"])

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
            examples_multiple[c].append([r[0], f"[{', '.join(df.columns[1:][r[1:] == 1])}]"])

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


def llm_client(endpoint, api_key, base_url="", api_version=""):
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
            azure_endpoint=base_url,
            api_version=api_version,
        )

    if endpoint == "ollama":
        return OpenAI(api_key=api_key, base_url=base_url)

    if endpoint == "groq":
        return Groq(api_key=api_key)


def client_invoke_sensor(
    endpoint_name,
    model_name,
    client,
    df,
    SystemPrompt,
    Sensors,
    examples_txt,
    UserPrompt,
    temperature=0.0,
    max_tokens=4096,
    seed=42,
    response_format=None,
):
    """Call client for chat completion"""

    results = []
    responses = []

    for instance in tqdm(df.iterrows(), total=df.shape[0]):

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
        result["true_target_sensor"] = (
            instance[1].iloc[1:][instance[1].iloc[1:] == 1].index.to_list()
        )

        # add user prompt
        messages.append(
            {"role": "user", "content": UserPrompt.format(req=result["requirement"])}
        )

        # run LLM
        start_time = time.perf_counter()

        if endpoint_name == "azure":
            response = client.beta.chat.completions.parse(
                model=model_name,
                messages=messages,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        else:

            # add format response to prompt
            messages[0]["content"] = messages[0]["content"].replace(
                "</Solution Plan>",
                f"""4. Format the response in JSON format.
</Solution Plan>

<Output Format>
The JSON object must use the schema: {json.dumps(response_format.model_json_schema(), indent=2)}
</Output Format>""",
        )

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )

        response_time = round(time.perf_counter() - start_time, 6)
        result["ai_response"] = response.choices[0].message.content

        response_json = json.loads(result["ai_response"])

        result["pred_target_sensor"] = [
            key for key, value in response_json.items() if value == 1
        ]
        result["response_time"] = response_time

        result["accuracy"] = sorted(result["pred_target_sensor"]) == sorted(
            result["true_target_sensor"]
        )

        response_usage = response.usage.to_dict()
        if endpoint_name == "azure":
            response_usage.pop("completion_tokens_details")
            response_usage.pop("prompt_tokens_details")

        result.update(response_usage)

        results.append(result)
        responses.append(response)

    return results, responses


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


def invoke_bulk_sensor(
    model_name,
    client,
    batch,
    SystemPrompt,
    Sensors,
    examples_txt,
    UserPromptBulk,
    req_text,
    temperature=0.0,
    seed=42,
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
        seed=seed,
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


def client_invoke_actuator_sensor(
    endpoint_name,
    client,
    df,
    model_name,
    SensorActuator,
    examples_text,
    response_format,
    temperature=0.0,
    seed=42,
    max_tokens=256,
):
    results = []
    responses = []

    # for instance in tqdm(df.iloc[:5,:].iterrows(), total=df.shape[0]):
    for instance in tqdm(df.iterrows(), total=df.shape[0]):

        # system prompt
        messages = [
            {
                "role": "system",
                "content": SensorActuator.format(
                    examples=examples_text,
                    requirement=instance[1].iloc[0],
                ),
            }
        ]

        # run LLM
        result = {
            "requirement": instance[1].iloc[0],
            "target_actuator": instance[1].iloc[1],
            "model": model_name,
        }
        start_time = time.perf_counter()

        if endpoint_name == "azure":
            response = client.beta.chat.completions.parse(
                model=model_name,
                messages=messages,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        else:

            # add format response to prompt
            messages[0]["content"] = messages[0]["content"].replace(
                "</Solution Plan>",
                f"""5. Format the response in JSON format.
</Solution Plan>

<Output Format>
The JSON object must use the schema: {json.dumps(response_format.model_json_schema(), indent=2)}
</Output Format>""",
            )

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )

        response_time = round(time.perf_counter() - start_time, 6)
        result["ai_response"] = response.choices[0].message.content
        messages.append(
            {
                "role": "assistant",
                "content": result["ai_response"],
            }
        )

        result_json = json.loads(result["ai_response"])
        result["ai_answer"] = result_json["target_actuator"]
        result["response_time"] = response_time
        result["accuracy"] = instance[1].iloc[1] == result_json["target_actuator"]

        response_usage = response.usage.to_dict()
        if endpoint_name == "azure":
            response_usage.pop("completion_tokens_details")
            response_usage.pop("prompt_tokens_details")

        result.update(response_usage)

        results.append(result)
        responses.append(response)

    return results, responses


def calc_stats(results):
    accuracy = 0
    total_tokens = 0
    total_completion_tokens = 0
    total_time = 0

    for r in results:
        accuracy += r["accuracy"]
        total_tokens += r["total_tokens"]
        total_completion_tokens += r["completion_tokens"]
        total_time += r["response_time"]

    number_of_reqs = len(results)
    accuracy /= len(results)
    avg_time_per_req = round(total_time / len(results), 6)
    avg_token_per_req = total_tokens / len(results)
    avg_completion_token_per_req = total_completion_tokens / len(results)

    return (
        number_of_reqs,
        accuracy,
        avg_time_per_req,
        avg_token_per_req,
        avg_completion_token_per_req,
        total_tokens,
        total_completion_tokens,
        total_time,
    )


def save_responses(base_path, prefix, **kwargs):
    # save results
    time = dt.now()

    results_path = "results/{prefix}_{model}_n-{examples}_acc-{accuracy}_{time}.json"
    results_path = results_path.format(
        prefix=prefix,
        model=kwargs["model_name"],
        examples=kwargs["n_examples"],
        time=time.strftime("%m.%d.%Y-%H:%M:%S"),
        accuracy=round(kwargs["accuracy"], 3),
    )

    results_file = base_path / results_path
    results_file.parent.mkdir(exist_ok=True)
    results_file.touch()

    with results_file.open("w") as f:
        json.dump(
            {
                "accuracy": kwargs["accuracy"],
                "number_of_reqs": kwargs["number_of_reqs"],
                "total_tokens": kwargs["total_tokens"],
                "total_completion_tokens": kwargs["total_completion_tokens"],
                "avg_token_per_req": kwargs["avg_token_per_req"],
                "avg_completion_token_per_req": kwargs["avg_completion_token_per_req"],
                "avg_time_per_req": kwargs["avg_time_per_req"],
                "examples": kwargs["examples"],
                "responses": kwargs["results"],
            },
            f,
            indent=4,
        )

    return results_file

import numpy as np
import time
import json
from datetime import datetime as dt
from openai import AzureOpenAI, OpenAI
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
            examples_multiple[c].append(
                [r[0], f"[{', '.join(df.columns[1:][r[1:] == 1])}]"]
            )

        # add both single and multiple into one place
        examples.update(examples_multiple)

        indexes_to_drop = np.array(indexes_to_drop).flatten()

    return indexes_to_drop, examples


def get_example_txt(df_examples, N_EXAMPLES=1):

    if N_EXAMPLES > 1:
        indexes_to_drop, examples = get_examples_from_df(
            df_examples.drop(columns=["selected"]), N_EXAMPLES
        )

    if N_EXAMPLES == 1:  # pick pre selected example when N_EXAMPLES = 1
        df_examples_n1 = df_examples[df_examples["selected"] == 1].copy()
        df_examples_n1.drop(columns=["selected"], inplace=True)
        indexes_to_drop, examples = get_examples_from_df(df_examples_n1, 1)

    # join all examples in a text format to add to prompt
    examples_txt = ""

    for e1 in examples.values():
        for e2 in e1:
            examples_txt += f"Requirement: {e2[0]}\n"
            examples_txt += f"Target Sensor/s: {e2[1]}\n"
            examples_txt += "\n"

    return examples_txt, examples


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

    if endpoint == "novita":
        return OpenAI(api_key=api_key, base_url=base_url)


def invoke_client(
    endpoint_name,
    model_name,
    client,
    messages,
    temperature,
    seed,
    max_tokens,
    response_format,
):

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

    elif endpoint_name == "novita":

        response_format_json = response_format.model_json_schema()
        response_format_novita = {"type": "json_schema", "json_schema": {"name": response_format_json.pop("title")}}
        response_format_novita["json_schema"]["schema"] = response_format_json

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            response_format=response_format_novita,
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

    return response, response_time


def client_invoke_sensor(
    endpoint_name,
    model_name,
    client,
    df,
    SystemPrompt,
    Sensors,
    examples_txt,
    UserPrompt,
    limit_instances=0,
    temperature=0.0,
    max_tokens=4096,
    seed=42,
    response_format=None,
    rpm_limit=0,
):
    """Call client for chat completion"""

    results = []

    total_time = 0

    if limit_instances > 0:
        df_iter = df.iloc[:limit_instances, :].iterrows()
    else:
        df_iter = df.iterrows()

    for i, instance in tqdm(enumerate(df_iter, start=1), total=df.shape[0]):

        # sleep when rpm_limit for the rest of the minute + 2
        if rpm_limit > 0:
            if i % rpm_limit == 0:
                sleep_time = round((total_time / 60) % 1 * 60, 0) + 2
                print(f"Sleeping for {sleep_time} ...")
                time.sleep(sleep_time)

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
        response, response_time = invoke_client(
            endpoint_name,
            model_name,
            client,
            messages,
            temperature,
            seed,
            max_tokens,
            response_format,
        )
        total_time += response_time

        result["ai_response"] = response.choices[0].message.content

        try:
            response_json = json.loads(result["ai_response"])
            result["is_parsed"] = True

            result["pred_target_sensor"] = [
                key for key, value in response_json.items() if value == 1
            ]

            result["accuracy"] = sorted(result["pred_target_sensor"]) == sorted(
                result["true_target_sensor"]
            )
        except:
            result["pred_target_sensor"] = []
            result["is_parsed"] = False
            result["accuracy"] = False

        result["response_time"] = response_time

        response_usage = response.usage.to_dict()
        if endpoint_name == "azure":
            response_usage.pop("completion_tokens_details")
            response_usage.pop("prompt_tokens_details")

        result.update(response_usage)

        results.append(result)

    return results


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


def requirement_text_bulk(batches, columns):
    # split and format batches
    for i, batch in enumerate(batches):
        idx = batch.index.to_list()
        req = batch["requirement"].tolist()
        vec = batch.apply(
            lambda row: [col for col in columns[1:] if row[col] == 1], axis=1
        ).tolist()
        batches[i] = (idx, req, vec)

    # requirement text
    req_texts = []
    for batch in batches:
        req_text = ""
        ix, req, vec = batch
        for i in range(len(ix)):
            req_text += f"<ID> {ix[i]} <Text> {req[i]}.\n"
        req_texts.append(req_text)

    return batches, req_texts


def invoke_bulk_sensor(
    endpoint_name,
    model_name,
    client,
    batches,
    req_texts,
    examples_txt,
    SystemPrompt,
    Sensors,
    UserPromptBulk,
    limit_batch_size=0,
    temperature=0.0,
    seed=42,
    max_tokens=4096,
    response_format=None,
    rpm_limit=0,
):
    results = []

    if limit_batch_size > 0:
        batch_iter = zip(batches[:limit_batch_size], req_texts)
    else:
        batch_iter = zip(batches, req_texts)

    total_time = 0

    for batch_id, (batch, req_text) in tqdm(
        enumerate(batch_iter, start=1), total=len(batches)
    ):

        # sleep when rpm_limit for the rest of the minute + 2
        if rpm_limit > 0:
            if batch_id % rpm_limit == 0:
                sleep_time = round((total_time / 60) % 1 * 60, 0) + 2
                print(f"Sleeping for {sleep_time} ...")
                time.sleep(sleep_time)

        messages = [
            {
                "role": "system",
                "content": SystemPrompt.format(sensors=Sensors, examples=examples_txt),
            }
        ]

        # add user prompt
        messages.append(
            {"role": "user", "content": UserPromptBulk.format(req=req_text)}
        )

        response, response_time = invoke_client(
            endpoint_name,
            model_name,
            client,
            messages,
            temperature,
            seed,
            max_tokens,
            response_format,
        )

        response_json = json.loads(response.choices[0].message.content)

        for ix in range(len(batch[0])):
            result = {"batch_id": batch_id}

            for resp_req in response_json["requirements"]:
                if resp_req["req_id"] == batch[0][ix]:
                    result["id"] = resp_req["req_id"]
                    result["requirement"] = resp_req["text"]
                    result["true_vector"] = batch[2][ix]
                    result["pred_vector"] = [
                        key
                        for key, value in resp_req["target_sensor"].items()
                        if value == 1
                    ]
                    result["accuracy"] = result["true_vector"] == result["pred_vector"]
                    result["ai_response"] = response.choices[0].message.content
                    result["response_time"] = response_time

                    usage_dict = response.usage.to_dict().copy()
                    if endpoint_name == "azure":
                        usage_dict.pop("prompt_tokens_details")
                        usage_dict.pop("completion_tokens_details")

                    result.update(usage_dict)

                    results.append(result)

            total_time += response_time

    return results


def client_invoke_actuator_sensor(
    endpoint_name,
    client,
    df,
    model_name,
    SensorActuator,
    examples_text,
    response_format,
    limit_instances=0,
    temperature=0.0,
    seed=42,
    max_tokens=256,
    rpm_limit=0,
):
    results = []

    total_time = 0

    if limit_instances > 0:
        df_iter = df.iloc[:limit_instances, :].iterrows()
    else:
        df_iter = df.iterrows()

    for i, instance in tqdm(enumerate(df_iter, start=1), total=df.shape[0]):

        # sleep when rpm_limit for the rest of the minute + 2
        if rpm_limit > 0:
            if i % rpm_limit == 0:
                sleep_time = round((total_time / 60) % 1 * 60, 0) + 2
                print(f"Sleeping for {sleep_time} ...")
                time.sleep(sleep_time)

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
        response, response_time = invoke_client(
            endpoint_name,
            model_name,
            client,
            messages,
            temperature,
            seed,
            max_tokens,
            response_format,
        )
        total_time += response_time

        result["ai_response"] = response.choices[0].message.content
        messages.append(
            {
                "role": "assistant",
                "content": result["ai_response"],
            }
        )
        result["response_time"] = response_time
        # try exception for parsing
        try:
            result_json = json.loads(result["ai_response"])
            result["ai_answer"] = result_json["target_actuator"]
            result["accuracy"] = instance[1].iloc[1] == result_json["target_actuator"]
            result["is_parsed"] = True
        except:
            print(result["ai_response"])
            result["ai_answer"] = ""
            result["accuracy"] = False
            result["is_parsed"] = False

        response_usage = response.usage.to_dict()
        if endpoint_name == "azure":
            response_usage.pop("completion_tokens_details")
            response_usage.pop("prompt_tokens_details")

        result.update(response_usage)

        results.append(result)

    return results


def calc_stats(results, **kwargs):
    accuracy = 0
    total_tokens = 0
    total_completion_tokens = 0
    total_time = 0

    for r in results:
        accuracy += r["accuracy"]
        total_tokens += r["total_tokens"]
        total_completion_tokens += r["completion_tokens"]
        total_time += r["response_time"]

    number_of_requests = (
        kwargs["number_of_requests"]
        if kwargs.get("number_of_requests") is not None
        else len(results)
    )
    accuracy /= len(results)
    avg_time_per_req = round(total_time / len(results), 6)
    avg_token_per_req = total_tokens / len(results)
    avg_completion_token_per_req = total_completion_tokens / len(results)

    return (
        number_of_requests,
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

    data = {
        "accuracy": kwargs["accuracy"],
        "batch_size": kwargs.get("batch_size"),
        "number_of_requests": kwargs["number_of_requests"],
        "number_of_examples": kwargs["n_examples"],
        "total_tokens": kwargs["total_tokens"],
        "total_completion_tokens": kwargs["total_completion_tokens"],
        "avg_token_per_req": kwargs["avg_token_per_req"],
        "avg_completion_token_per_req": kwargs["avg_completion_token_per_req"],
        "avg_time_per_req": kwargs["avg_time_per_req"],
        "examples": kwargs["examples"],
        "responses": kwargs["results"],
    }

    with results_file.open("w") as f:
        json.dump(data, f, indent=4)

    return results_file

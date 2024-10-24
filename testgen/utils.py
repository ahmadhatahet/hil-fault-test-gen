import re
import numpy as np

def get_examples_from_df(df, n_examples):
    
    # collect used examples to exclude from test dataset
    indexes_to_drop = []

    # collect examples with single fault
    examples = {}

    for c in df.columns[1:]:
        if df[c].sum() == 0: continue
        examples[c] = []
        i = df.loc[ (df[c] == 1) & (df[df.columns[1:][df.columns[1:]!=c]].sum(axis=1) == 0) ].sample(n_examples)
        indexes_to_drop.append(i.index)
        for e in i.values:
            examples[c].append([e[0], "[" + ",".join(map(str, e[1:])) + "]"])
            
    # collect examples with multiple faults
    examples_multiple = {}

    t = df.loc[df[df.columns[1:]].sum(axis=1) == 2]
    idx = t.sample(n_examples).index
    indexes_to_drop.append(idx)
    t = df.iloc[idx, :].values

    for r in t:
        c = "&".join(df.columns[1:][r[1:]==1])
        if examples_multiple.get(c) is None:
            examples_multiple[c] = []
        examples_multiple[c].append([r[0], "[" + ",".join(map(str, r[1:])) + "]"])
        
    # add both single and multiple into one place
    examples.update(examples_multiple)

 
    indexes_to_drop = np.array(indexes_to_drop).flatten()

    return indexes_to_drop, examples



def invoke_instance(llm, df, SystemPrompt, Sensors, examples_txt, UserPrompt,instance):
    
    # system prompt
    messages = [
        {'role': 'system',
        'content': SystemPrompt.format(sensors=Sensors,examples=examples_txt)}
    ]

    result = {}

    result["idx"] = instance[0]
    result["requirement"] = instance[1].iloc[0]
    result["true_vector"] = "[" + ",".join(map(str, instance[1].iloc[1:])) + "]"


    # add user prompt
    messages.append({"role":"user", "content":UserPrompt.format(req=result["requirement"])})

    # run LLM
    response = llm.invoke(messages)
    result["ai_response"] = response.content
    result["pred_vector"] = parse_result(result["ai_response"])

    result["accuracy"] = result["pred_vector"] == result["true_vector"]

    result.update(response.response_metadata["token_usage"])

    return result, response

def parse_result(res):
    '''Parse the LLM result'''
    pattern = r"\[(.*?)\]"
    vec = re.findall(pattern, res)[0]
    return f"[{vec}]".replace(" ", "")

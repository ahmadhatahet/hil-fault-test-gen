import pandas as pd
import numpy as np
import json
from testgen.prompts import Sensors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_sensors_labels():
    labels = Sensors.split("\n")
    for i, sensor in enumerate(labels):
        labels[i] = sensor[sensor.find("(") + 1 : sensor.find(")")].strip()

    return np.array(labels)


def get_best_results(results_path, file_pattern):
        
    available_results = list(results_path.glob(file_pattern))

    # drop file with lower accuracy if multiple are found
    available_results_dict = {}

    for i, f in enumerate(available_results):
        t_ = f.name.split("_acc-")
        if available_results_dict.get(t_[0]) is None:
            available_results_dict[t_[0]] = (i, float(t_[1].split("_")[0]))
        else:
            if available_results_dict[t_[0]][1] < float(t_[1].split("_")[0]):
                del available_results[available_results_dict[t_[0]][0]]
                available_results_dict[t_[0]] = (i, float(t_[1].split("_")[0]))

    return available_results


def calc_scores(df):
    df_scores = pd.DataFrame(
        columns=["sensor", "accuracy", "precision", "recall", "f1"]
    )

    y_true = df["true_label"]
    y_pred = df["pred_label"]
    unique_labels = np.unique(y_true)
    unique_labels.sort()

    accuracy_score_ = round(accuracy_score(y_true, y_pred), 2)
    precision_score_ = round(
        precision_score(y_true, y_pred, average="weighted", labels=unique_labels), 2
    )
    recall_score_ = round(
        recall_score(y_true, y_pred, average="weighted", labels=unique_labels), 2
    )
    f1_score_ = round(
        f1_score(y_true, y_pred, average="weighted", labels=unique_labels), 2
    )

    df_scores.loc[df_scores.shape[0] + 1] = [
        "All",
        accuracy_score_,
        precision_score_,
        recall_score_,
        f1_score_,
    ]

    for label in unique_labels:
        t_ = df[df["true_label"] == label]

        y_true_ = t_["true_label"]
        y_pred_ = t_["pred_label"]

        accuracy_score_ = round(accuracy_score(y_true_, y_pred_), 2)
        precision_score_ = round(
            precision_score(y_true_, y_pred_, average="weighted", labels=[label]), 2
        )
        recall_score_ = round(
            recall_score(y_true_, y_pred_, average="weighted", labels=[label]), 2
        )
        f1_score_ = round(
            f1_score(y_true_, y_pred_, average="weighted", labels=[label]), 2
        )

        df_scores.loc[df_scores.shape[0] + 1] = [
            label,
            accuracy_score_,
            precision_score_,
            recall_score_,
            f1_score_,
        ]

    return df_scores.set_index("sensor")



def data_to_df_single(data, model_name, type_, number_examples):
    # split responses and general stats
    responses = pd.DataFrame(data["responses"])
    responses.set_index("idx", inplace=True)
    
    responses.insert(0, "model", model_name)
    responses.insert(1, "type", type_)
    responses.insert(2, "n_examples", number_examples)
    
    return responses


def data_to_df_bulk(data, model_name, type_, number_examples):
    
    type_, type_n = type_.split("-")
    type_n = int(type_n)

    # split responses and general stats
    responses = pd.DataFrame()
    for res in data["responses"]:
        responses = pd.concat([responses, pd.DataFrame(res)])
    responses.set_index("idx", inplace=True)

    responses.insert(0, "model", model_name)
    responses.insert(1, "type", type_)
    responses.insert(2, "type_n", type_n)
    responses.insert(3, "n_examples", number_examples)

    
    return responses


def result_to_df(labels, filename, results_path):

    type_, model_name, number_examples, *_ = filename.stem.split("_")
    number_examples = int(number_examples.split("-")[-1])

    file_under_investigation = results_path / filename
    with file_under_investigation.open("r") as f:
        data = json.load(f)

    # convert responses to df if single or bulk
    if len(type_.split("-")) == 1: # means single not bulk
        responses = data_to_df_single(data, model_name, type_, number_examples)
    else:
        responses = data_to_df_bulk(data, model_name, type_, number_examples)
    
    
    # collect label names for predictions and ground truth
    responses["true_label"] = responses["true_vector"].map(lambda x: " & ".join(labels[np.array(x[1:-1].split(",")).astype(bool)]) )
    responses["pred_label"] = responses["pred_vector"].map(lambda x: " & ".join(labels[np.array(x[1:-1].split(",")).astype(bool)]) )
    
    del data["responses"]
    
    return responses, data



def analyze(labels, filename, results_path):

    type_, model_name, number_examples, *_ = filename.stem.split("_")
    number_examples = int(number_examples.split("-")[-1])
    responses, data = result_to_df(labels, filename, results_path)
    responses.drop(columns=["model", "type", "n_examples"], inplace=True)

    # collect stats per experiment
    stats = pd.DataFrame({k: [v]for k,v in data.items()})
    stats.insert(0, "model_name", model_name)
    stats.insert(1, "number_examples", number_examples)
    
    
    # accuracy per sensor
    summarize_ = responses.groupby(["true_label"]).aggregate({
        "accuracy": "sum",
        "true_label": "count"
    })

    summarize_.columns = ["true", "total"]
    summarize_["false"] = summarize_["total"] - summarize_["true"]


    all_vals = summarize_.sum(axis=0).values.tolist()
    summarize_.loc["All"] = all_vals

    summarize_.insert(0, "model_name", model_name)
    summarize_.insert(1, "number_examples", number_examples)
    
    
    # convert responses to df if single or bulk
    if len(type_.split("-")) == 1: # means single not bulk
        stats.insert(0, "type", type_)
        summarize_.insert(0, "type", type_)
    else:
        type_, type_n = type_.split("-")
        type_n = int(type_n)
        stats.insert(0, "type", type_)
        stats.insert(1, "type_n", type_n)
        summarize_.insert(0, "type", type_)
        summarize_.insert(1, "type_n", type_n)

    df_scores = calc_scores(responses)

    summarize_ = summarize_.merge(df_scores, left_index=True, right_index=True).reset_index(names="sensors")
    
    return stats, summarize_


def analyze_multiple(labels, results, results_path):
    stats = pd.DataFrame()
    summarize = pd.DataFrame()
    
    for filename in results:
        stats_, summarize_ = analyze(labels, filename, results_path)
        
        stats = pd.concat([stats, stats_])
        summarize = pd.concat([summarize, summarize_])
        
    stats.drop(columns="examples", inplace=True)
    summarize.reset_index(drop=True, inplace=True)
    
    type_, *_ = filename.stem.split("_")
    if len(type_.split("-")) == 1: # means single not bulk
        stats.set_index("number_examples", inplace=True)
    else:
        stats.set_index("type_n", inplace=True)

    return stats, summarize 
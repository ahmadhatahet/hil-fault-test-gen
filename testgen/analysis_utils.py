import pandas as pd
import numpy as np
import json
from testgen.prompts import Sensors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings

warnings.filterwarnings("ignore")


def get_sensors_labels():
    labels_t = Sensors.split("\n")
    labels = {}
    for i, sensor in enumerate(labels_t):
        pos_1 = sensor.find("(")
        labels[sensor[: pos_1 - 1].strip().replace(" ", "_").lower()] = sensor[
            pos_1 + 1 : sensor.find(")")
        ].strip()

    return labels


def get_best_results(results_path, file_pattern):
    available_results = list(results_path.glob(file_pattern))

    # drop file with lower accuracy if multiple are found
    available_results_dict = {}

    for i, f in enumerate(available_results):
        _, _, model_name, n, acc, time_ = f.name.split("_")

        n = int(n.split("-")[1])
        acc = float(acc.split("-")[1])
        time_ = time_[:-5]

        model_name = f"{model_name}_{n}"

        if available_results_dict.get(model_name) is None:
            available_results_dict[model_name] = [n, acc]
        else:
            if (available_results_dict[model_name][0] == n) and (
                available_results_dict[model_name][1] <= acc
            ):
                available_results.pop(i)
                available_results_dict[model_name] = [n, acc]
    return available_results, available_results_dict


def calc_scores(responses_df):

    df_scores = pd.DataFrame()

    for model in responses_df["model"].unique():
        for n_e in responses_df[(responses_df["model"] == model)][
            "number_of_examples"
        ].unique():

            df_t = responses_df[
                (responses_df["model"] == model)
                & (responses_df["number_of_examples"] == n_e)
            ]

            df_scores_t = pd.DataFrame(
                columns=["sensor", "accuracy", "precision", "recall", "f1"]
            )

            y_true = df_t["pred_target_sensor_str"]
            y_pred = df_t["true_target_sensor_str"]
            unique_labels = np.unique(y_true)
            unique_labels.sort()

            # remove from unique_labels the labels that are empty
            unique_labels = unique_labels[unique_labels != ""]

            accuracy_score_ = round(accuracy_score(y_true, y_pred), 2)
            precision_score_ = round(
                precision_score(
                    y_true, y_pred, average="weighted", labels=unique_labels
                ),
                2,
            )
            recall_score_ = round(
                recall_score(y_true, y_pred, average="weighted", labels=unique_labels),
                2,
            )
            f1_score_ = round(
                f1_score(y_true, y_pred, average="weighted", labels=unique_labels), 2
            )

            df_scores_t.loc[df_scores_t.shape[0] + 1] = [
                "All",
                accuracy_score_,
                precision_score_,
                recall_score_,
                f1_score_,
            ]

            for label in unique_labels:
                t_ = df_t[df_t["pred_target_sensor_str"] == label]

                y_true_ = t_["pred_target_sensor_str"]
                y_pred_ = t_["true_target_sensor_str"]

                accuracy_score_ = round(accuracy_score(y_true_, y_pred_), 2)
                precision_score_ = round(
                    precision_score(
                        y_true_,
                        y_pred_,
                        average="weighted",
                        labels=[label],
                    ),
                    2,
                )
                recall_score_ = round(
                    recall_score(
                        y_true_,
                        y_pred_,
                        average="weighted",
                        labels=[label],
                    ),
                    2,
                )
                f1_score_ = round(
                    f1_score(y_true_, y_pred_, average="weighted", labels=[label]), 2
                )

                df_scores_t.loc[df_scores_t.shape[0] + 1] = [
                    label,
                    accuracy_score_,
                    precision_score_,
                    recall_score_,
                    f1_score_,
                ]

            df_scores_t.insert(0, "model", model)
            df_scores_t.insert(1, "number_of_examples", n_e)

            df_scores = pd.concat([df_scores, df_scores_t.set_index("sensor")])

    return df_scores


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


def result_to_df(results):

    stats_df = pd.DataFrame()
    mismatch_stats_df = pd.DataFrame()
    responses_df = pd.DataFrame()

    for res in results:

        stats = json.loads(res.read_text())
        filename = res.stem
        is_bulk = filename.find("_b-") > -1

        if is_bulk:
            _, category, model_name, n, b, acc, time_ = filename.split("_")
            b = int(b.split("-")[1])
        else:
            _, category, model_name, n, acc, time_ = filename.split("_")

        n = int(n.split("-")[1])
        acc = float(acc.split("-")[1])
        time_ = time_[:-5]

        del stats["examples"]

        responses = stats.pop("responses")
        responses = pd.DataFrame(responses).set_index("idx")
        responses.insert(0, "model", model_name)
        responses.insert(1, "number_of_examples", n)

        stats.pop("", n)

        stats = pd.DataFrame.from_dict(stats, orient="index").T
        # drop "number_of_examples" if exists
        if "number_of_examples" in stats.columns:
            stats.drop(columns="number_of_examples", inplace=True)
        stats.insert(0, "model", model_name)
        stats.insert(1, "number_of_examples", n)
        stats["time"] = time_

        exploded_responses = (
            responses[responses["accuracy"] == False]
            .explode("true_target_sensor")
            .explode("pred_target_sensor")
        )
        mismatch_stats = (
            exploded_responses.groupby("true_target_sensor")["pred_target_sensor"]
            .value_counts(dropna=False)
            .unstack(fill_value=0)
        )
        mismatch_stats.reset_index(drop=False, inplace=True)
        mismatch_stats.columns.name = ""
        mismatch_stats.insert(0, "model", model_name)
        mismatch_stats.insert(1, "number_of_examples", n)

        if is_bulk:
            stats.insert(2, "batch_size", b)
            responses.insert(2, "batch_size", b)
            mismatch_stats.insert(2, "batch_size", b)

        stats_df = pd.concat([stats_df, stats], ignore_index=True)
        mismatch_stats_df = pd.concat(
            [mismatch_stats_df, mismatch_stats], ignore_index=True
        )
        responses_df = pd.concat([responses_df, responses], ignore_index=True)

    return stats_df, responses_df, mismatch_stats_df


def analyze(labels, filename, results_path):

    type_, model_name, number_examples, *_ = filename.stem.split("_")
    number_examples = int(number_examples.split("-")[-1])
    responses, data = result_to_df(labels, filename, results_path)
    responses.drop(columns=["model", "type", "n_examples"], inplace=True)

    # collect stats per experiment
    stats = pd.DataFrame({k: [v] for k, v in data.items()})
    stats.insert(0, "model_name", model_name)
    stats.insert(1, "number_examples", number_examples)

    # accuracy per sensor
    summarize_ = responses.groupby(["true_label"]).aggregate(
        {"accuracy": "sum", "true_label": "count"}
    )

    summarize_.columns = ["true", "total"]
    summarize_["false"] = summarize_["total"] - summarize_["true"]

    all_vals = summarize_.sum(axis=0).values.tolist()
    summarize_.loc["All"] = all_vals

    summarize_.insert(0, "model_name", model_name)
    summarize_.insert(1, "number_examples", number_examples)

    # convert responses to df if single or bulk
    if len(type_.split("-")) == 1:  # means single not bulk
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

    summarize_ = summarize_.merge(
        df_scores, left_index=True, right_index=True
    ).reset_index(names="sensors")

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
    if len(type_.split("-")) == 1:  # means single not bulk
        stats.set_index("number_examples", inplace=True)
    else:
        stats.set_index("type_n", inplace=True)

    return stats, summarize

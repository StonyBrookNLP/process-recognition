import argparse

import numpy as np
import pandas as pd

import config
import utils

__author__ = 'chetannaik'


def get_column_scores(adf, experiment):
    mean_score = 0
    max_score = 0
    median_score = 0
    for score in config.SCORES:
        score_lst = np.array(
            map(lambda x: float(x), adf[score].dropna().tolist()))
        score_lst = sorted(score_lst, reverse=True)
        if experiment == "SRLQADSv2Top5" and len(score_lst) > 5:
            score_lst = score_lst[:5]

        if len(score_lst) > 0:
            mean_score += np.mean(score_lst)
            max_score += np.max(score_lst)
            median_score += np.median(score_lst)

    return mean_score, max_score, median_score


def get_row_scores(adf, experiment):
    ans_scores = []
    for label, row in adf.iterrows():
        tmp_score = 0
        for score in config.SCORES:
            if not np.isnan(float(row[score])):
                tmp_score += row[score]
        ans_scores.append(tmp_score)

    score_lst = filter(lambda x: x > 0.0, ans_scores)
    score_lst = sorted(score_lst, reverse=True)
    if experiment == "SRLQADSv2Top5" and len(score_lst) > 5:
        score_lst = score_lst[:5]

    mean_score = np.mean(score_lst) if (sum(score_lst) + len(
        score_lst)) > 0 else 0
    max_score = np.max(score_lst) if (sum(score_lst) + len(
        score_lst)) > 0 else 0
    median_score = np.median(score_lst) if (sum(score_lst) + len(
        score_lst)) > 0 else 0
    return mean_score, max_score, median_score


def count_scores(row):
    scores = row[config.SCORES].tolist()
    return len([s for s in scores if (not np.isnan(s) and s > 0.05)])


def sum_scores(row):
    scores = row[config.SCORES].tolist()
    return sum([s for s in scores if (not np.isnan(s) and s > 0.05)])


def get_max_role_scores(adf):
    adf["NUM_SCORES"] = adf.apply(count_scores, axis=1)
    adf["SUM_SCORES"] = adf.apply(sum_scores, axis=1)
    ndf = adf.groupby("NUM_SCORES")
    s_groups = ndf.groups.keys()
    group_scores = {}
    for s_group in s_groups:
        sdf = ndf.get_group(s_group)
        group_max = sdf.SUM_SCORES.max()
        group_scores[s_group] = group_max
    final_score = 0
    for g, g_score in group_scores.iteritems():
        final_score += g_score * config.GROUP_WEIGHTS[g]
    return final_score, final_score, final_score


def aggregate_scores(experiment):
    df = pd.read_csv("output/" + experiment + "/features.tsv", sep="\t")
    df_columns = ["QUESTION", "ANSWER_CHOICE", "MEAN_SCORE", "MAX_SCORE",
                  "MEDIAN_SCORE", "CORRECT_ANSWER"]
    result_df = pd.DataFrame(columns=df_columns)
    dfq = df.groupby("QUESTION")
    questions = dfq.groups.keys()

    for qid, question in enumerate(questions):
        qdf = dfq.get_group(question)
        dfa = qdf.groupby("ANSWER_CHOICE")
        answers = dfa.groups.keys()
        for answer in answers:
            adf = dfa.get_group(answer)
            if config.SCORE_TYPE == "COLUMN_SCORE":
                mean_scr, max_scr, median_scr = get_column_scores(adf,
                                                                  experiment)
            elif config.SCORE_TYPE == "ROW_SCORE":
                mean_scr, max_scr, median_scr = get_row_scores(adf, experiment)
            elif config.SCORE_TYPE == "MAX_ROLE_SCORE":
                mean_scr, max_scr, median_scr = get_max_role_scores(adf)

            temp_df = pd.DataFrame([[question, answer, mean_scr, max_scr,
                                     median_scr,
                                     adf["CORRECT_ANSWER"].tolist()[0]]],
                                   columns=df_columns)
            result_df = pd.concat([result_df, temp_df])
    result_df = result_df.reset_index(drop=True)
    result_df.to_csv("output/" + experiment + "/combined_scores.tsv", sep="\t")
    print ("\nGenerated output/" + experiment
           + "/combined_scores.tsv containing aggregated scores.")


def main(experiment):
    utils.generate_experiment_scores(experiment)
    aggregate_scores(experiment)
    df = pd.read_csv("output/" + experiment + "/combined_scores.tsv", sep="\t",
                     index_col=0)
    questions = list(set(df.QUESTION.tolist()))
    df_columns = ["QUESTION", "MEAN_PREDICTION", "MAX_PREDICTION",
                  "MEDIAN_PREDICTION", "CORRECT_ANSWER"]
    pred_df = pd.DataFrame(columns=df_columns)

    for question in questions:
        qdf = df.groupby("QUESTION").get_group(question)
        correct_ans = qdf.CORRECT_ANSWER.any()
        mean_prediction = ""
        max_prediction = ""
        median_prediction = ""
        best_mean = -float("inf")
        best_max = -float("inf")
        best_median = -float("inf")

        for idx, row in qdf.iterrows():
            if row["MEAN_SCORE"] > best_mean:
                best_mean = row["MEAN_SCORE"]
                mean_prediction = row["ANSWER_CHOICE"]
            if row["MAX_SCORE"] > best_max:
                best_max = row["MAX_SCORE"]
                max_prediction = row["ANSWER_CHOICE"]
            if row["MEDIAN_SCORE"] > best_median:
                best_median = row["MEDIAN_SCORE"]
                median_prediction = row["ANSWER_CHOICE"]
        temp_df = pd.DataFrame([[question, mean_prediction, max_prediction,
                                 median_prediction, correct_ans]],
                               columns=df_columns)
        pred_df = pd.concat([pred_df, temp_df])
    pred_df = pred_df.reset_index(drop=True)
    pred_df.to_csv("output/" + experiment + "/predictions.tsv", sep="\t")
    print ("Generated output/" + experiment
           + "/predictions.tsv containing predictions.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Frame alignment for QA.')
    parser.add_argument('--experiment', help='experiment to run')

    args = parser.parse_args()
    main(args.experiment)

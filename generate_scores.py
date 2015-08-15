import os
import argparse
import itertools

import numpy as np
import pandas as pd

import config
import entailment
import utils

__author__ = 'chetannaik'


def get_question_group_key(question_group, question):
    for q in question_group.groups.keys():
        if question.strip() in q.strip():
            return q
    return None


def get_question_frames(question, question_frames, experiment):
    """Question frame extractor.

    Args:
        question: A string representing the question without options.
        question_frames: Contents of question_frames.tsv file.

    Returns: A list of python dictionaries containing the the frames for the
        question extracted from question_frames.
    """
    q_group = question_frames.groupby('QUESTION')
    if len(filter(bool, set(question.strip().split('.')))) > 1:
        q_sentences = set(question.strip().split('.'))
        q_sentences = filter(bool, map(lambda x: x.strip(), q_sentences))
    else:
        q_sentences = []
    q_sentences.append(question.strip())
    dfs = []
    for q in q_sentences:
        key = get_question_group_key(q_group, q)
        if key:
            df = q_group.get_group(key)
            dfs.append(df)
    ret_df = pd.concat(dfs)
    ret_df = ret_df[config.ROLES[experiment]]
    ret_df.drop_duplicates(inplace=True)
    return ret_df.to_dict(orient='records')


def get_answer_group_key(process_group, process):
    for group in process_group.groups.keys():
        names = group.split(" | ")
        if utils.get_lemma(process) in names:
            return group
    return None


def get_answer_frames(answer, process_db, experiment):
    """Answer frame extractor.

    Args:
        answer: A string representing the answer choice.
        process_db: Contents of process_frames.tsv file.


    Returns: a list of python dictionaries, containing frames for the answer
        with frame elements extracted from process_db.
    """
    p_group = process_db.groupby('PROCESS')
    key = get_answer_group_key(p_group, answer)
    if key:
        afs = p_group.get_group(key)
        afs = afs[config.ROLES[experiment]]
        afs.drop_duplicates(inplace=True)
        return afs.to_dict(orient='records')
    else:
        return []


def get_alignment_data(question_frames, answer_choices, process_db, experiment):
    """Gets alignment scores by calling aligner.

    Args:
        question_frames: A list of python dictionaries containing question
            frames.
        answer_choices: A python list containing answer choices.
        process_db:  Contents of process_frames.tsv file.

    Returns: A dictionary for each answer choice containing a list of
        dictionaries frame roles, role elements and their entailment scores.
    """
    answer_data = dict()
    for answer in answer_choices:
        answer_frames = get_answer_frames(answer, process_db, experiment)
        answer_scores = aligner(question_frames, answer_frames, experiment)
        answer_data[answer] = answer_scores
    return answer_data


def get_frame_directional_score(question_frame, answer_frame, experiment):
    frame_scores = dict()
    temp_scores = dict()
    temp_scores["FORWARD"] = dict()
    for frame_element in config.ROLES[experiment]:
        q_element = question_frame[frame_element]
        a_element = answer_frame[frame_element]
        if pd.isnull(q_element) or pd.isnull(a_element):
            score = np.NaN
        else:
            ret = entailment.get_ai2_textual_entailment(a_element, q_element)
            a_scores = map(lambda x: x['score'], ret['alignments'])
            if len(a_scores):
                mean_a_score = np.mean(a_scores)
            else:
                mean_a_score = 0

            confidence = ret['confidence'] if ret['confidence'] else 0
            score = mean_a_score * confidence
        temp_scores["FORWARD"][frame_element] = (q_element, a_element, score)

    temp_scores["BACKWARD"] = dict()
    for frame_element in config.ROLES[experiment]:
        q_element = question_frame[frame_element]
        a_element = answer_frame[frame_element]
        if pd.isnull(q_element) or pd.isnull(a_element):
            score = np.NaN
        else:
            ret = entailment.get_ai2_textual_entailment(q_element, a_element)
            a_scores = map(lambda x: x['score'], ret['alignments'])
            if len(a_scores):
                mean_a_score = np.mean(a_scores)
            else:
                mean_a_score = 0

            confidence = ret['confidence'] if ret['confidence'] else 0
            score = mean_a_score * confidence
        temp_scores["BACKWARD"][frame_element] = (q_element, a_element, score)

    best_score = 0
    best_direction = None
    for k, v in temp_scores.iteritems():
        direction = k
        score = sum(map(lambda x: x[2], v.values()))
        if score >= best_score:
            best_score = score
            best_direction = direction

    for frame_element in config.ROLES[experiment]:
        q_element = question_frame[frame_element]
        a_element = answer_frame[frame_element]
        if pd.notnull(q_element) and pd.notnull(a_element):
            q_list = map(lambda x: x.strip(), q_element.split("|"))
            a_list = map(lambda x: x.strip(), a_element.split("|"))
            if utils.has_filter_keyword(q_list):
                frame_scores[frame_element] = ("", a_element, np.NaN)
            elif utils.has_filter_keyword(a_list):
                frame_scores[frame_element] = (q_element, "", np.NaN)
            else:
                frame_scores[frame_element] = \
                    temp_scores[best_direction][frame_element]
        elif pd.notnull(q_element):
            frame_scores[frame_element] = (q_element, "", score)
        elif pd.notnull(a_element):
            frame_scores[frame_element] = ("", a_element, score)
        else:
            frame_scores[frame_element] = ("", "", np.NaN)
    # frame_scores[utils.DEFINITION] = answer_frame[utils.DEFINITION]
    return frame_scores


def get_role_directional_score(question_frame, answer_frame, experiment):
    frame_scores = dict()
    for frame_element in config.ROLES[experiment]:
        q_element = question_frame[frame_element]
        a_element = answer_frame[frame_element]
        if pd.isnull(q_element) or pd.isnull(a_element):
            score = np.NaN
        else:
            ret = entailment.get_ai2_textual_entailment(
                utils.remove_filter_words(a_element),
                utils.remove_filter_words(q_element))
            a_scores = map(lambda x: x['score'], ret['alignments'])
            if len(a_scores):
                mean_a_score = np.mean(a_scores)
            else:
                mean_a_score = 0

            confidence = ret['confidence'] if ret['confidence'] else 0
            score1 = mean_a_score * confidence
            ret = entailment.get_ai2_textual_entailment(
                utils.remove_filter_words(q_element),
                utils.remove_filter_words(a_element))
            a_scores = map(lambda x: x['score'], ret['alignments'])
            if len(a_scores):
                mean_a_score = np.mean(a_scores)
            else:
                mean_a_score = 0

            confidence = ret['confidence'] if ret['confidence'] else 0
            score2 = mean_a_score * confidence
            score = max(score1, score2)
        if pd.notnull(q_element) and pd.notnull(q_element):
            if q_element in ["process", "processes"]:
                frame_scores[frame_element] = ("", a_element, np.NaN)
            else:
                frame_scores[frame_element] = (q_element, a_element, score)
        elif pd.notnull(q_element):
            frame_scores[frame_element] = (q_element, "", score)
        elif pd.notnull(a_element):
            frame_scores[frame_element] = ("", a_element, score)
        else:
            frame_scores[frame_element] = ("", "", np.NaN)
            #     frame_scores[utils.DEFINITION] = answer_frame[utils.DEFINITION]
    return frame_scores


def aligner(question_frames, answer_frames, experiment):
    """Aligns a question frame with a answer frame and calls entailment service
    to get a match score.

    Args:
        question_frames: A list of python dictionaries containing question
            frames.
        answer_frames: A list of python dictionaries containg answer frame
            elements.

    Returns: A number representing the match score of question frame with all
        the answer frames.
    """
    answer_scores = []
    for question_frame in question_frames:
        for answer_frame in answer_frames:
            if config.SCORE_DIRECTION_ABSTRACTION == "FRAME":
                frame_scores = get_frame_directional_score(question_frame,
                                                           answer_frame,
                                                           experiment)
            else:
                frame_scores = get_role_directional_score(question_frame,
                                                          answer_frame,
                                                          experiment)
            answer_scores.append(frame_scores)
    return answer_scores


def process_shard(shard_experiment):
    shard_num, experiment = shard_experiment

    process_db = pd.read_csv(
        "data/" + experiment + "/frames.cv." + str(shard_num) + ".tsv",
        sep='\t')
    process_db.PROCESS = process_db.PROCESS.str.lower()

    questions = pd.read_csv(
        "data/" + experiment + "/question.list.cv." + str(shard_num) + ".tsv",
        sep='\t')

    question_frames = pd.read_csv(
        "data/" + experiment + "/question.framepredict.cv." + str(
            shard_num) + ".tsv", sep='\t')

    ret_list = []
    for qid, row in questions.iterrows():
        if qid > 0 and (qid + 1) % 60 == 0:
            print "."
        else:
            print ".",
        question = row.QUESTION
        q_frames = get_question_frames(question, question_frames, experiment)
        answer_choices = filter(pd.notnull, row[config.OPTIONS].tolist())
        answer_data = get_alignment_data(q_frames, answer_choices, process_db,
                                         experiment)
        correct_answer = row[row.ANSWER]

        for answer, data in answer_data.iteritems():
            for roles in data:
                out_row = [question, answer]  # add roles[DEFINITION] here
                contents = map(lambda x: list(roles[x]),
                               config.ROLES[experiment])
                contents = list(itertools.chain.from_iterable(contents))
                out_row.extend(contents)
                out_row.append(correct_answer)
                ret_list.append(out_row)
    return ret_list


def main(num_shards, experiment):
    utils.get_filter_words()
    if not os.path.exists("output/" + experiment):
        os.makedirs("output/" + experiment)

    result = []
    for shard_num in range(num_shards):
        result.extend(process_shard((shard_num, experiment)))
    result_df = pd.DataFrame(result)
    col_list = ['QUESTION', 'ANSWER_CHOICE']  # add 'ANSWER_SENTENCE' here
    contents = map(lambda r: ['Q_' + r, 'A_' + r, r + '_SCORE'],
                   config.ROLES[experiment])
    contents = list(itertools.chain.from_iterable(contents))
    col_list.extend(contents)
    col_list.append('CORRECT_ANSWER')
    result_df.columns = col_list
    result_df.to_csv("output/" + experiment + "/features.tsv", sep="\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Frame alignment for QA.')
    parser.add_argument('--num_shards', type=int,
                        help='number of shards/folds')
    parser.add_argument('--experiment', help='experiment to run')

    args = parser.parse_args()
    main(args.num_shards, args.experiment)

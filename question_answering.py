from nltk.stem.wordnet import WordNetLemmatizer
import entailment

import csv
import numpy as np
import operator

# To access process_frames read from 'process_frames.tsv' and
# 'question_frames.tsv'
PROCESS = 0
UNDERGOER = 1
CAUSE = 2
MANNER = 3
RESULT = 4
SOURCE = 5
DEFINITIONS = 6
FRAME_ELEMENTS = [UNDERGOER, CAUSE, MANNER, RESULT]

# To access questions read from 'questions.tsv' and 'question_frames.tsv'
QUESTION = 0
OPTIONS = 1
OPTION_A = 2
OPTION_B = 3
OPTION_C = 4
OPTION_D = 5
QUESTION_PROCESS_NAME = 6
ANSWER = 7

# TODO: Store the above mapping in a dictionary with key in string format
ANS_MAP = {"OPTION_A": 2,
           "OPTION_B": 3,
           "OPTION_C": 4,
           "OPTION_D": 5}


def read_tsv(filename):
    try:
        fh = open(filename, 'rb')
        reader = csv.reader(fh, delimiter='\t')
        header = reader.next()
        contents = []
        for row in reader:
            contents.append(row)
    except IOError:
        print "Error: cannot read the file."
    finally:
        fh.close()
    return header, contents


def get_processes(reader):
    """Returns all the processes in the process_frames.tsv file."""
    processes = set(map(lambda row: row[PROCESS].lower(), reader))
    return processes


def get_question_frame(question, question_frames):
    """Question frame extractor.

    Args:
        question: A string representing the question without options.
        question_frames: Contents of question_frames.tsv file.

    Returns: A python dictionary q_frame for the question with
        frame elements extracted from question_frames.
    """
    q_frame = dict()
    for row in question_frames:
        if row[QUESTION] == question:
            q_frame[UNDERGOER] = row[UNDERGOER]
            q_frame[CAUSE] = row[CAUSE]
            q_frame[MANNER] = row[MANNER]
            q_frame[RESULT] = row[RESULT]
    return q_frame


def get_answer_frames(answer, process_db):
    """Answer frame extractor.

    Args:
        answer: A string representing the answer choice.
        process_db: Contents of process_frames.tsv file.

    Returns: a answer_frames (a list) of python dictionaries,
        containing frames for the answer with frame elements
        extracted from process_db.
    """
    wnl = WordNetLemmatizer()
    answer_frames = list()
    for row in process_db:
        answer_frame = dict()
        if wnl.lemmatize(answer) in row[PROCESS].lower():
            answer_frame[UNDERGOER] = row[UNDERGOER]
            answer_frame[CAUSE] = row[CAUSE]
            answer_frame[MANNER] = row[MANNER]
            answer_frame[RESULT] = row[RESULT]
            answer_frames.append(answer_frame)
    return answer_frames


def ranker(question_frame, answer_choices, process_db):
    """Ranks the answer_choices by calling aligner.

    Args:
        question_frame: A python dictionary containg question frame elements.
        answer_choices: A python list containing answer choices.
        process_db:  Contents of process_frames.tsv file.

    Returns: A ranked list of tuples containing answer choices and their
        scores.
    """
    answer_scores = dict()
    for answer in answer_choices:
        answer_frames = get_answer_frames(answer, process_db)
        frame_score = aligner(question_frame, answer_frames)
        answer_scores[answer] = frame_score
    ranked_answers = sorted(answer_scores.items(), key=operator.itemgetter(1),
                            reverse=True)
    return ranked_answers


def aligner(question_frame, answer_frames):
    """Aligns a question frame with a answer frame and calls entailment service
    to get a match score.

    Args:
        question_frame: A python dictionary containg question frame elements.
        answer_frames: A list of python dictionaries containg answer frame
            elements.

    Returns: A number representing the match score of question frame with all
        the answer frames.
    """
    answer_scores = []
    for answer_frame in answer_frames:
        frame_scores = dict()
        for frame_element in FRAME_ELEMENTS:
            q_element = question_frame[frame_element]
            a_element = answer_frame[frame_element]
            ret = entailment.get_ai2_textual_entailment(a_element, q_element)
            if ret['confidence'] is None:
                score = 0
            else:
                score = ret['confidence']
            frame_scores[frame_element] = (q_element, a_element, score)
        answer_scores.append(frame_scores)
    answer_score = list()
    for frame_score in answer_scores:
        answer_score.append(sum(map(lambda x: x[2], frame_score.values())))
    score = np.sum(filter(lambda x: x > 0.0, answer_score))
    return score


def main():
    _, process_db = read_tsv('process_frames.tsv')
    _, questions = read_tsv('questions.tsv')
    _, question_frames = read_tsv('question_frames.tsv')

    fh = open("output.tsv", "wt")
    row_string = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"
    header = row_string.format("QUESTION", "OPTION_A", "OPTION_B", "OPTION_C",
                               "OPTION_D", "CORRECT_ANSWER",
                               "PREDICTED_ANSWER", "SCORES")
    fh.write(header)

    for num, row in enumerate(questions):
        question = row[QUESTION]
        q_frame = get_question_frame(question, question_frames)
        answer_choices = [row[OPTION_A], row[OPTION_B],
                          row[OPTION_C], row[OPTION_D]]
        ranked_answers = ranker(q_frame, answer_choices, process_db)
        p_answer, confidence = ranked_answers[0]
        if confidence <= 0.0:
            p_answer = "Don't know :("

        out_row = row_string.format(question, row[OPTION_A], row[OPTION_B],
                                    row[OPTION_C], row[OPTION_D],
                                    row[ANS_MAP[row[ANSWER]]], p_answer,
                                    ranked_answers)

        fh.write(out_row)
    fh.close()
    print "Done!"


if __name__ == '__main__':
    main()

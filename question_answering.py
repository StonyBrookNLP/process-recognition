from nltk.stem.wordnet import WordNetLemmatizer
import entailment

import csv
import logging
import numpy as np
import operator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log file handler
handler = logging.FileHandler('qa.log')
handler.setLevel(logging.INFO)

# Logging format
formatter = logging.Formatter('%(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# To access process_frames read from 'process_frames.tsv' and
# 'question_frames.tsv'
PROCESS = 0
UNDERGOER = 1
ENABLER = 2
TRIGGER = 3
RESULT = 4
UNDERSPECIFIED = 5
FRAME_ELEMENTS = [UNDERGOER, ENABLER, TRIGGER, RESULT, UNDERSPECIFIED]

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
    processes = set()
    for row in reader:
        processes.add(row[PROCESS])
    processes = [process.lower() for process in processes]
    return processes


def clean_string(entry):
    wnl = WordNetLemmatizer()
    return wnl.lemmatize(entry.strip().lower())


def get_question_frames(question, question_frames):
    """Question frame extractor.

    Args:
        question: A string representing the question without options.
        question_frames: Contents of question_frames.tsv file.

    Returns: A python dictionary q_frame for the question with
        frame elements extracted from question_frames.
    """
    q_sentences = set(question.strip().split('.'))
    q_sentences.add(question.strip())
    q_frames = list()
    for row in question_frames:
        q_frame = dict()
        if any(row[QUESTION].strip() in q for q in q_sentences):
            q_frame[UNDERGOER] = row[UNDERGOER]
            q_frame[ENABLER] = row[ENABLER]
            q_frame[TRIGGER] = row[TRIGGER]
            q_frame[RESULT] = row[RESULT]
            q_frame[UNDERSPECIFIED] = row[UNDERSPECIFIED]
            q_frames.append(q_frame)
    return q_frames


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
            answer_frame[ENABLER] = row[ENABLER]
            answer_frame[TRIGGER] = row[TRIGGER]
            answer_frame[RESULT] = row[RESULT]
            answer_frame[UNDERSPECIFIED] = row[UNDERSPECIFIED]
            answer_frames.append(answer_frame)
    return answer_frames


def ranker(question_frames, answer_choices, process_db):
    """Ranks the answer_choices by calling aligner.

    Args:
        question_frames: A python list containing question frame elements.
        answer_choices: A python list containing answer choices.
        process_db:  Contents of process_frames.tsv file.

    Returns: A ranked list of tuples containing answer choices and their
        scores.
    """
    answer_scores = dict()
    for answer in answer_choices:
        logger.info("Answer: %s", answer)
        answer_frames = get_answer_frames(answer, process_db)
        frame_score = aligner(question_frames, answer_frames)
        answer_scores[answer] = frame_score
        logger.info("----------\n")
    ranked_answers = sorted(answer_scores.items(), key=operator.itemgetter(1),
                            reverse=True)
    return ranked_answers


def aligner(question_frames, answer_frames):
    """Aligns a question frame with a answer frame and calls entailment service
    to get a match score.

    Args:
        question_frames: A list of python dictionaries containg question frame
            elements.
        answer_frames: A list of python dictionaries containg answer frame
            elements.

    Returns: A number representing the match score of question frame with all
        the answer frames.
    """
    answer_scores = []
    for question_frame in question_frames:
        for answer_frame in answer_frames:
            frame_scores = dict()
            for frame_element in FRAME_ELEMENTS:
                q_element = question_frame[frame_element]
                a_element = answer_frame[frame_element]
                ret = entailment.get_ai2_textual_entailment(
                    a_element, q_element)
                if ret['confidence'] is None:
                    score = 0
                else:
                    score = ret['confidence']
                frame_scores[frame_element] = (q_element, a_element, score)
            answer_scores.append(frame_scores)
    # logging loop
    for a in answer_scores:
        for k, v in a.iteritems():
            logger.info("%s: %s", k, v)
        logger.info("--")
    answer_score = list()
    for f_s in answer_scores:
        answer_score.append(sum(map(lambda x: x[2], f_s.values())))
    scores = filter(lambda x: x > 0.0, answer_score)
    score = np.mean(scores) if (sum(scores) + len(scores)) > 0 else 0
    logger.info("SCORES: %s", answer_score)
    logger.info("MEAN:%s", score)
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
        q_frames = get_question_frames(question, question_frames)
        answer_choices = [row[OPTION_A], row[OPTION_B],
                          row[OPTION_C], row[OPTION_D]]
        logger.info("%s. %s", num + 1, question)
        logger.info("A: %s", row[OPTION_A])
        logger.info("B: %s", row[OPTION_B])
        logger.info("C: %s", row[OPTION_C])
        logger.info("D: %s\n\n", row[OPTION_D])
        ranked_answers = ranker(q_frames, answer_choices, process_db)
        p_answer, confidence = ranked_answers[0]
        if confidence <= 0.0:
            p_answer = "Don't know :("
            logger.info("Predicted Answer: Don't know")
        else:
            logger.info("Predicted Answer: %s with %s confidence", p_answer,
                        confidence)
        logger.info("Correct Answer: %s\n", row[ANS_MAP[row[ANSWER]]])
        logger.info("Match Scores: %s\n\n", ranked_answers)
        logger.info("=========================")
        out_row = row_string.format(question, row[OPTION_A], row[OPTION_B],
                                    row[OPTION_C], row[OPTION_D],
                                    row[ANS_MAP[row[ANSWER]]], p_answer,
                                    ranked_answers)

        fh.write(out_row)
    fh.close()
    print "Done!"


if __name__ == '__main__':
    main()

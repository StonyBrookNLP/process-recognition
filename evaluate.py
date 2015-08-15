import argparse

import pandas as pd

__author__ = 'chetannaik'


def main(experiment):
    df = pd.read_csv("output/" + experiment + "/predictions.tsv", sep="\t",
                     index_col=0)
    total = 0
    num_mean = 0
    num_max = 0
    num_median = 0

    for label, row in df.iterrows():
        total += 1
        if row["MEAN_PREDICTION"] == row["CORRECT_ANSWER"]:
            num_mean += 1
        if row["MAX_PREDICTION"] == row["CORRECT_ANSWER"]:
            num_max += 1
        if row["MEDIAN_PREDICTION"] == row["CORRECT_ANSWER"]:
            num_median += 1

    with open("output/" + experiment + "/evaluation.txt", "w") as text_file:
        text_file.write("RESULTS:\n")
        text_file.write("\nTotal number of questions: {}\n".format(total))
        text_file.write("\nnum answered (mean)  : {}\n".format(num_mean))
        text_file.write("num answered (max)   : {}\n".format(num_max))
        text_file.write("num answered (median): {}\n".format(num_median))
        text_file.write("\n% answered (mean)  : {}\n".format(
            round(num_mean * 100 / float(total), 2)))
        text_file.write("% answered (max)   : {}\n".format(
            round(num_max * 100 / float(total), 2)))
        text_file.write("% answered (median): {}\n".format(
            round(num_median * 100 / float(total), 2)))

    print "\nRESULTS:\n"
    print "Total number of questions: ", total

    print "\nnum answered (mean)  : ", num_mean
    print "num answered (max)   : ", num_max
    print "num answered (median): ", num_median

    print "\n% answered (mean)  : ", round(num_mean * 100 / float(total), 2)
    print "% answered (max)   : ", round(num_max * 100 / float(total), 2)
    print "% answered (median): ", round(num_median * 100 / float(total), 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Frame alignment for QA.')
    parser.add_argument('--experiment', help='experiment to run')

    args = parser.parse_args()
    main(args.experiment)

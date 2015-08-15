import time

import generate_scores
import rank_answers
import evaluate
import analysis
import config

__author__ = 'chetannaik'


def main():
    print config.SCORE_TYPE

    for experiment in config.EXPERIMENTS:
        now = time.time()
        print "\nEXPERIMENT: {}\n".format(experiment)
        generate_scores.main(config.SHARDS[experiment], experiment)
        rank_answers.main(experiment)
        evaluate.main(experiment)
        analysis.main(experiment)
        lstring = 'experiment: {}\ntime = {} sec'
        print lstring.format(str(experiment), str(time.time() - now))
    print "Done!"


if __name__ == '__main__':
    main()

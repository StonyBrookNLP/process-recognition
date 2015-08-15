__author__ = 'chetannaik'


# EXPERIMENTS = ["SRLManual", "SRLQA", "SRLQADSv1", "SRLQADSv2", "SRLQADSv2Top5", "MATE_AutoTrigger", "MATE_ManualTrigger", "SRLManualExtended"]
EXPERIMENTS = ["SRLManual"]

SHARDS = {"SRLManual": 1,
          "SRLManualExtended": 1,
          "SRLQA": 5,
          "SRLQADSv1": 5,
          "SRLQADSv2": 5,
          "SRLQADSv2Top5": 5,
          "MATE_AutoTrigger": 1,
          "MATE_ManualTrigger": 1}

# SCORE_TYPES = ["ROW_SCORE", "COLUMN_SCORE", "MAX_ROLE_SCORE"]
SCORE_TYPE = "ROW_SCORE"

# SCORE_DIRECTION_ABSTRACTION = ["FRAME", "ROLE"]
SCORE_DIRECTION_ABSTRACTION = "ROLE"

GROUP_WEIGHTS = {5: 1, 4: 0.8, 3: 0.24, 2: 0.072, 1: 0.0216, 0: 0}

SCORES = []

BASIC_ROLES = ['UNDERGOER', 'ENABLER', 'TRIGGER', 'RESULT']

EXTENDED_ROLES = ['UNDERGOER', 'ENABLER', 'TRIGGER', 'THEME', 'RESULT',
                  'MEDIUM', 'SOURCE', 'TARGET', 'LOCATION', 'DIRECTION']

ROLES = {"SRLManual": BASIC_ROLES,
         "SRLManualExtended": EXTENDED_ROLES,
         "SRLQA": BASIC_ROLES,
         "SRLQADSv1": BASIC_ROLES,
         "SRLQADSv2": BASIC_ROLES,
         "SRLQADSv2Top5": BASIC_ROLES,
         "MATE_AutoTrigger": BASIC_ROLES,
         "MATE_ManualTrigger": BASIC_ROLES}

OPTIONS = ['OPTION_A', 'OPTION_B', 'OPTION_C', 'OPTION_D']

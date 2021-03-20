# this file serves as a dump for global constants

# lists of target variables
TARGETS_REGRESSION = ['OS months', 'PMFS Time to endpoint  PM or no PM']
TARGETS_MULTICLASS =['If OM_number of lesions at last FU']
TARGETS_BINARY = ['PM definition  ≥4 lesions', 'PMFS Oligo Status (≤5) mantained until  last FU 0=Y', '≤10', 'By feasibility']


# the features used in the project - by commenting them in and out you can set which ones to use
# TODO: Marius, please double check this list is correct
FEATURE_LIST = [
    "DoB",
    "Gender",
    "Primary Tumour",
    "First Met Organ Site",
    "CTV cc",
    "SUVmax Baseline PET-CT",
    # 'Local Relapse Y(1) /N(0)',
    # 'LRFS Months',
    # 'Progression Elsewhere(Y:1 / N: 0)',
    "Same organ (0:Y 1:N 2:Both)",
    "Systemic Tx (0 =no or pre, 1= combination with during or post)",
    "First lesion(s) SDRT Only (1=Y)",
    "SOMA            1=Y 0=N",
    "Number of targets at 1st Tx",
    "Overall Regimen",
    "N LRR",
    "Patient with LF with rescue",
    "Tumour Burden 1st Tx cc",
    "Tumour burden         I SOMA",
    "Largest single OM burden",
    #'SOMA > 1st OM  0=N 1=Y',
    "DFS months between repeat Tx",
    "Highest SUVmax at 1st Tx",
    "N. of  target organs at 1st Tx"
    # 'N of targets    I SOMA',
    # 'Interval between ablations',
    # 'Δ Tumour burden	Min Burden',
    # 'Max Burden',
    # 'Average burden',
    # 'Min %Δ Tumour burden',
    # 'Max %Δ Tumour burden',
    # 'Mean %Δ Tumour burden',
    # 'Min N',
    # 'Max N',
    # 'Min SOMA interval',
    # 'Max SOMA Interval',
    # 'Average SOMA Interval',
    # 'Cumulative Tumour burden',
    # 'Largest single SOMA burden',
    # 'Highest SUVmax ever',
    # 'N. of  target organs involved in total',
    # 'Repeat TX',
    # 'Total number of targets',
    # 'Total SOMA lesions'
]

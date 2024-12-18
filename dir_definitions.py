import os

# main folders
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
CODE_DIR = os.path.join(ROOT_DIR, 'python_code')
RESOURCES_DIR = os.path.join(ROOT_DIR, 'resources')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

# subfolders
CONFIG_PATH = os.path.join(CODE_DIR, 'config.yaml')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
CONFIG_RUNS_DIR = os.path.join(RESOURCES_DIR, 'config_runs')
ECC_MATRICES_DIR = os.path.join(RESOURCES_DIR, 'ECC_matrices')
BP_WEIGHTS = os.path.join(RESOURCES_DIR, 'bp_weights')

import os


# Directories
MAIN_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(MAIN_DIR, os.pardir))

INPUT    = os.path.join(MAIN_DIR, 'input')
OUTPUT   = os.path.join(MAIN_DIR, 'output')
RESULT   = os.path.join(MAIN_DIR, 'result')
TRAINED_MODELS = os.path.join(MAIN_DIR, 'trained_models')
CONFIG   = os.path.join(MAIN_DIR, 'config')

# Identifiers for unique environments
ENVIRONMENT_NAMES = ['Env']

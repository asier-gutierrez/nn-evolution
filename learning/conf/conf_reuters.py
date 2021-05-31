from collections import defaultdict

SEED = 42
DROPOUT_SEED = 42
ANALYSIS_TYPES = [
    defaultdict(list, {'name': 'LAYER_SIZE', 'values': [4, 16, 32, 128, 256]}),
    defaultdict(list, {'name': 'NUMBER_LAYERS', 'values': [2, 4, 6, 8, 10]}),
    defaultdict(list, {'name': 'INPUT_ORDER', 'values': [1, 2, 3, 4, 5]}),
    defaultdict(list, {'name': 'NUMBER_LABELS', 'values': [2, 6, 12, 23, 46]}),
    defaultdict(list, {'name': 'DROPOUT', 'values': [0.0, 0.2, 0.4, 0.5, 0.8]}),
    defaultdict(list, {'name': 'LEARNING_RATE', 'values': [0.1, 0.01, 0.001, 0.0001, 0.00001]})
]

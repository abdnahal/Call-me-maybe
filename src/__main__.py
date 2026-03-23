import argparse
from typing import List, Any


def argparser() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition",
                        default='data/input/functions_definition.json')
    parser.add_argument('--input',
                        default='data/input/function_calling_tests.json')
    parser.add_argument('--output', default='data/output/function_calls.json')
    args = parser.parse_args()
    return args

import argparse

from contraction._lib import runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--flag', default=False, action='store_true', help="Test flag.")
    return parser.parse_args()


def run() -> None:
    _args = parse_args()

    runner.run()

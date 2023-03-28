"""Main script to build all instruction datasets.

Usage:
    python build_instruction_datasets.py --datasets BrazilianBarExam BrCAD5
"""
import argparse
import functools
import logging
import multiprocessing
from typing import Optional, Sequence, Type

from abstract_dataset import AbstractDataset
from datasets import ALL_DATASETS
from datasets import DATASETS_ALREADY_BUILT
from datasets import ERRONEOUS_DATASETS


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Builds the instruction datasets', )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Builds a small version of the dataset for debugging',
    )
    parser.add_argument(
        '--build_from_scratch',
        action='store_true',
        help='Builds the dataset from scratch, even if it already exists',
    )
    parser.add_argument("--processes",
                        type=int,
                        default=1,
                        help="Number of processes to use")
    parser.add_argument("--datasets",
                        type=str,
                        nargs="+",
                        default=[],
                        help="Datasets to build (default: all)")
    args = parser.parse_args(args)

    # If no datasets are specified, build all of them
    if not args.datasets:
        args.datasets = sorted(dataset.__name__ for dataset in ALL_DATASETS)
    # Get the actual classes for each named dataset.
    all_datasets = {dataset.__name__: dataset for dataset in ALL_DATASETS}
    args.datasets = [all_datasets[dataset] for dataset in args.datasets]

    return args


def _build_dataset(dataset: Type[AbstractDataset], debug_size: int) -> None:
    # TODO(arya): We have a Liskov substitution principle violation here.
    #   AbstractDataset's __init__ takes two arguments, but the
    #  __init__s of the subclasses take none. Should rectify, perhaps by
    #  composition instead of inheritance.
    dataset().build_instruction_dataset(debug_size=debug_size)


def build_instruction_datasets(datasets: Sequence[Type[AbstractDataset]],
                               *,
                               processes: int,
                               debug: bool = False,
                               build_from_scratch: bool = False) -> None:
    if debug:
        datasets_to_build = ERRONEOUS_DATASETS
        debug_size = 5
        processes = 1  # Parallelism would only introduce more confusion.
    else:
        datasets_to_build = set(datasets) - ERRONEOUS_DATASETS
        debug_size = -1

        if not build_from_scratch:
            datasets_to_build = datasets_to_build - DATASETS_ALREADY_BUILT

    logging.info("Building datasets: %s",
                 [d.__name__ for d in datasets_to_build])

    build_one = functools.partial(_build_dataset, debug_size=debug_size)

    with multiprocessing.Pool(processes=processes) as pool:
        pool.map(build_one, datasets_to_build)


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    build_instruction_datasets(args.datasets,
                               processes=args.processes,
                               debug=args.debug,
                               build_from_scratch=args.build_from_scratch)

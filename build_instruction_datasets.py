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
import instruction_manager
from lawinstruct_datasets import ALL_DATASETS, LEGAL_DATASETS, NON_LEGAL_DATASETS
from lawinstruct_datasets import DATASETS_ALREADY_BUILT
from lawinstruct_datasets import ERRONEOUS_DATASETS


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Builds the instruction datasets')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Builds a small version of the dataset for debugging',
    )
    parser.add_argument(
        '--build_erroneous_datasets',
        action='store_true',
        help='Builds only the erroneous datasets for debugging',
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
                        default=["all"],
                        help="Datasets to build (default: `all`). `legal` and `nonlegal` are also valid options.")
    parser.add_argument("--language_mode",
                        choices=["english", "multilingual"],
                        default="english",
                        help="Languages for instructions (default: `english`)")
    parser.add_argument("--instruction_bank_size",
                        type=int,
                        default=10,
                        help="Number of instruction paraphrases to sample from")
    args = parser.parse_args(args)
    logging.info(f"args: {args!r}")

    # If no datasets are specified, build all of them
    if not args.datasets or args.datasets == ["all"]:
        args.datasets = sorted(dataset.__name__ for dataset in ALL_DATASETS)
    elif args.datasets == ["legal"]:
        args.datasets = sorted(dataset.__name__ for dataset in LEGAL_DATASETS)
    elif args.datasets == ["nonlegal"]:
        args.datasets = sorted(dataset.__name__ for dataset in NON_LEGAL_DATASETS)

    # Get the actual classes for each named dataset.
    dataset_lookup = {dataset.__name__: dataset for dataset in ALL_DATASETS}
    args.datasets = [dataset_lookup[dataset] for dataset in args.datasets]

    return args


def _build_dataset(
        dataset: Type[AbstractDataset],
        instructions: instruction_manager.InstructionManager,
        debug_size: int
) -> None:
    # TODO(arya): We have a Liskov substitution principle violation here.
    #   AbstractDataset's __init__ takes two arguments, but the
    #  __init__s of the subclasses take none. Should rectify, perhaps by
    #  composition instead of inheritance.
    dataset().build_instruction_dataset(
        instructions,
        debug_size=debug_size)


def build_instruction_datasets(
        datasets: Sequence[Type[AbstractDataset]],
        instructions: instruction_manager.InstructionManager,
        *,
        processes: int = -1,
        debug: bool = False,
        build_erroneous_datasets: bool = False,
        build_from_scratch: bool = False
) -> None:
    logging.info("Building instruction datasets: %s", datasets)
    if debug:
        debug_size = 5
        processes = 1  # Parallelism would only introduce more confusion.
    else:
        debug_size = -1
        processes = processes

    if build_erroneous_datasets:
        datasets_to_build = sorted(list(ERRONEOUS_DATASETS))
    else:
        datasets_to_build = [dataset for dataset in datasets if dataset not in ERRONEOUS_DATASETS]
        if not build_from_scratch:
            datasets_to_build = [dataset for dataset in datasets_to_build if dataset not in DATASETS_ALREADY_BUILT]

    logging.info("Building datasets: %s",
                 [d.__name__ for d in datasets_to_build])

    build_one = functools.partial(
        _build_dataset,
        instructions=instructions,
        debug_size=debug_size,
    )

    # with multiprocessing.Pool(processes=processes) as pool:
    #     pool.map(build_one, datasets_to_build)
    for dataset in datasets_to_build:
        build_one(dataset)

    # TODO add option to run only remaining datasets not yet saved in data folder


if __name__ == '__main__':
    args = parse_args()
    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(level=log_level)
    instructions = instruction_manager.InstructionManager(
        mode=args.language_mode,
        instruction_bank_size=args.instruction_bank_size)
    build_instruction_datasets(args.datasets,
                               instructions,
                               processes=args.processes,
                               debug=args.debug,
                               build_erroneous_datasets=args.build_erroneous_datasets,
                               build_from_scratch=args.build_from_scratch)

"""Translate instruction-tuning dataset's instructions into other languages."""
from collections.abc import Sequence
import json
import os
import pathlib

from absl import app
from absl import flags
from absl import logging
import datasets
import easynmt
import tqdm


_LANGUAGES = flags.DEFINE_multi_string(
    'languages',
    ['bg', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr', 'ga', 'hr',
     'hu', 'it', 'lt', 'lv', 'mt', 'nl', 'pl', 'pt', 'ro', 'sk', 'sl', 'sv'],
    'Languages to translate to.',
)
# for faster speed, use "opus-mt" or "m2m_100_418M", for better quality
# use "opus-mt" or "m2m_100_1.2B"
_MODEL_NAME = flags.DEFINE_string(
    'model_name',
    'm2m_100_1.2B',
    'Model name to use for translation.',
)
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size',
    16,
    'Batch size to use for translation inference.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    'data/translated/',
    'Directory to save translated data to.',
)


def translate_to_target_lang(
        model: easynmt.EasyNMT,
        dataset: datasets.Dataset,
        save_dir: pathlib.Path,
        source_lang: str,
        target_lang: str,
) -> None:
    texts: list[str] = []
    original_langs: list[str] = []

    result: dict[str, list[str]] = {}

    for group in tqdm.tqdm(dataset):
        result[group] = []
        for instruction in dataset[group]:
            translated = model.translate(
                instruction,
                source_lang=source_lang,
                target_lang=target_lang,
                batch_size=_BATCH_SIZE.value,
            )
            logging.info('Original: %r', instruction)
            logging.info('Translated: %r', translated)
            texts.append(translated)
            original_langs.append(source_lang)
            result[group].append(translated)

    # dataset = datasets.Dataset.from_dict({
    #     'text': texts,
    #     'translation_model': [_MODEL_NAME.value] * len(original_langs),
    #     'original_lang': original_langs,
    #     'target_lang': [target_lang] * len(original_langs),
    # })

    filepath = save_dir / f'{target_lang}.json'
    logging.info('Saving to %s', filepath)
    # dataset.to_json(
    #     filepath,
    #     force_ascii=False,
    #     orient='records',
    #     lines=True,
    # )
    with open(filepath, 'w', encoding='utf-8') as f:
        # Disabled ensure_ascii to save Unicode data without escapes.
        json.dump(result, f, indent=4, ensure_ascii=False)
    logging.info('Done saving to %s', filepath)
    # logging.info('Compressing...')
    # os.system(f'xz -zkf -T0 {filepath}')  # -T0 uses all cores
    # logging.info('Done compressing.')


def get_file(index: int) -> str:
    return f'https://huggingface.co/datasets/pile-of-law/lawnstruct/resolve/main/data/train.{index}.jsonl.xz'


def main(args: Sequence[str]) -> None:
    if len(args) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Prepare the output directory.
    save_dir = pathlib.Path(_OUTPUT_DIR.value)
    logging.info('Creating save directory %s', save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load the dataset.
    with open('instruction_prompts/en.json', 'r') as f:
        dataset = json.load(f)
    source_lang = 'en'
    print(dataset)
    for target_lang in _LANGUAGES.value:
        logging.info('Translating to %s', target_lang)
        model = easynmt.EasyNMT(_MODEL_NAME.value)
        translate_to_target_lang(model, dataset, save_dir, source_lang, target_lang)


if __name__ == '__main__':
    app.run(main)

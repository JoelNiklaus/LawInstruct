"""Translate instruction-tuning dataset's instructions into other languages."""
from collections.abc import Sequence
import json
import pathlib
from typing import Any

from absl import app
from absl import flags
from absl import logging
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
_OUTPUT_FILENAME = flags.DEFINE_string(
    'output_filename',
    'multilingual',
    'Save filename within output dir (.json extension will be added).'
)


def translate_to_target_langs(
        model: easynmt.EasyNMT,
        dataset: Any,
        source_lang: str,
        target_langs: Sequence[str],
) -> dict[str, dict[str, list[str]]]:
    """Translates text into many languages, producing an output file with same structure."""
    # Initialize result; structure is lang -> task_type -> instruction_text
    result: dict[str, dict[str, list[str]]] = {t: {} for t in target_langs}

    for target_lang in target_langs:
        for group in tqdm.tqdm(dataset):
            result[target_lang][group] = []
            for instruction in dataset[group]:
                translated = model.translate(
                    instruction,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    batch_size=_BATCH_SIZE.value,
                )
                logging.info('Original: %r', instruction)
                logging.info('Translated to %s: %r', target_lang, translated)
                result[target_lang][group].append(translated)

    return result

def main(args: Sequence[str]) -> None:
    if len(args) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Prepare the output directory.
    save_dir = pathlib.Path(_OUTPUT_DIR.value)
    logging.info('Creating save directory %r', save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{_OUTPUT_FILENAME}.json'
    del save_dir
    logging.info('Will save to %r', save_path)

    # Load the dataset.
    with open('instruction_prompts/en.json', 'r') as f:
        dataset = json.load(f)
    source_lang = 'en'
    logging.debug('The dataset: %s', dataset)
    target_languages = _LANGUAGES.value
    logging.info('Translating to %s', target_languages)
    model = easynmt.EasyNMT(_MODEL_NAME.value)
    # Translate the dataset.
    result = translate_to_target_langs(
        model,
        dataset[source_lang],
        source_lang,
        target_languages,
    )
    # Save the translations.
    logging.info('Saving to %s', save_path)
    with open(save_path, 'w', encoding='utf-8') as f:
        # Disabled ensure_ascii to save Unicode data without escapes.
        json.dump(result, f, indent=4, ensure_ascii=False)
    logging.info('Done saving to %s', save_path)


if __name__ == '__main__':
    app.run(main)

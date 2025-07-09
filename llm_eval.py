#!/usr/bin/env python3
"""
Evaluation script for compositional-generalization experiments that uses the
**llm** package (a modern Python LLM client) instead of Google's genai SDK.

Key changes
-----------
*   Uses `llm` package to interact with Gemini and other LLMs instead of Google's SDK.
*   Simplified LLM interaction with `model.prompt()` and schema-enforced JSON responses.
*   Supports seamless resumption of evaluation via CSV-based checkpointing.
*   Compatible with various LLM backends supported by the llm package.
*   Maintains the same underlying evaluation logic for accuracy assessment.
*   Requirements line now reads: `pip install llm pandas numpy tqdm pydantic python-dotenv`.
*   Evaluates ALL test data without seed filtering.

Run `python evaluate_generalization.py --help` for CLI usage.
mkdir -p ~/Desktop/compositional_results && find . -type f \( -name 'decoder_results.json' -o -path '*/gemini_eval_*/*.csv' -o -path '*/rankings/*.csv' \) -print0 | rsync -aR --from0 --files-from=- . ~/Desktop/compositional_results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import llm
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import Field
from pydantic import create_model
from tqdm import tqdm

from compositional_gen.config import AVAILABLE_MODELS
from compositional_gen.config import COMPONENT_CONFIGS
from compositional_gen.data import ComponentConfig

load_dotenv()

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

_ARTICLE_RE = re.compile(r'^(?:a|an|the)\s+', flags=re.I)


def strip_article(text: str) -> str:
  """Return text without a, an, the + trailing space; case-preserving."""
  return _ARTICLE_RE.sub('', text).strip()


class NumpyJSONEncoder(json.JSONEncoder):
  """
  Custom JSON encoder that handles NumPy data types.

  Converts NumPy integers, floats, and arrays to their Python equivalents
  for proper JSON serialization.
  """

  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    if isinstance(obj, np.floating):
      return float(obj)
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return super().default(obj)


def find_all_comp_setups(model_dir: Path) -> list[str]:
  """Return every sub-directory that contains dataset_with_paths.pkl."""
  return sorted(
    [p.name for p in model_dir.iterdir() if (p / 'dataset_with_paths.pkl').exists()]
  )


def needs_eval(model_dir: Path, comp_setup: str, gemini_tag: str) -> bool:
  """True if summary.json does not exist yet."""
  res_dir = model_dir / comp_setup / f'gemini_eval_{gemini_tag}'
  return not (res_dir / 'summary.json').exists()


class LatentAnalyzer:
  """Query Gemini with an image + prompt and parse the structured reply (via llm)."""

  def __init__(self, model_name: str = 'gemini-2.0-flash') -> None:
    self.model = llm.get_model(model_name)
    self.model_name = model_name
    logger.info('Using Gemini model: %s', model_name)

  def analyze_images(
    self,
    image_paths: List[Union[str, Path]],
    schema_class: Type[BaseModel],
    prompt_template: str,
    ground_truth_latents: Optional[List[np.ndarray]] = None,
    components: Optional[List[Any]] = None,
    append_to_csv: Optional[Path] = None,
  ) -> List[Optional[Dict[str, Any]]]:
    """Return one JSON dict per image (None on failure)."""

    prompt_system = prompt_template.strip()

    correct_count = 0
    total_processed = 0
    all_latents: List[np.ndarray] = []

    results: List[Optional[Dict[str, Any]]] = []
    for i, path in enumerate(tqdm(image_paths, desc='Analyzing images')):
      try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
          logger.error('File not found: %s', p)
          results.append(None)
          continue

        img_attachment = llm.Attachment(path=str(p))

        max_retries = 10
        for attempt in range(max_retries):
          try:
            resp = self.model.prompt(
              prompt_system,
              attachments=[img_attachment],
              schema=schema_class,
              temperature=0.0,
            )
            latent_dict = schema_class.model_validate_json(resp.text()).model_dump()
            break
          except Exception as e:
            if attempt < max_retries - 1:
              wait_time = 5
              logger.warning(
                'Exception: %s. Waiting %d s before retry %d/%d',
                e,
                wait_time,
                attempt + 1,
                max_retries,
              )
              time.sleep(wait_time)
              continue
            else:
              raise

        results.append(latent_dict)

        if append_to_csv is not None:
          first_write = not append_to_csv.exists()
          pd.DataFrame(
            [
              {
                'image_path': str(p),
                'gemini_json': json.dumps(latent_dict, cls=NumpyJSONEncoder),
              }
            ]
          ).to_csv(append_to_csv, mode='a', header=first_write, index=False)

        if ground_truth_latents is not None and components is not None:
          latent = latent_from_schema(latent_dict, components)
          all_latents.append(latent)

          binary_pred = (latent > 0.5).astype(int)
          gt = ground_truth_latents[i]

          if (binary_pred == gt).all():
            correct_count += 1

          total_processed += 1
          running_accuracy = correct_count / total_processed

          if total_processed % 1 == 0 or total_processed == len(image_paths):
            logger.info(
              f'Running accuracy: {correct_count}/{total_processed} '
              f'({running_accuracy:.2%})'
            )

      except Exception as exc:
        logger.error('Error analysing %s: %s', path, exc)
        results.append(None)

    # Final accuracy report (unchanged)
    if ground_truth_latents is not None and total_processed > 0:
      logger.info(
        f'Final accuracy: {correct_count}/{total_processed} '
        f'({correct_count/total_processed:.2%})'
      )

    return results


def create_schema_for_components(components: list[Any]) -> type[BaseModel]:
  boolean_mode = len(components) == 1 and not components[0].is_multi_component()

  fields: dict[str, tuple[type, Field]] = {}

  if boolean_mode:
    # one component, many boolean flags
    comp = components[0]
    for elem in comp.elements:
      base = strip_article(elem).replace('one ', '')
      clean = base.replace(' ', '_')
      fields[clean] = (
        bool,
        Field(..., description=f'Whether {elem} is in the image'),
      )
  else:
    for slot in components:
      fields[slot.name] = (
        str,
        Field(
          ...,
          description=f"{slot.name} - choose ONE of: {', '.join(slot.elements)}",
        ),
      )

  return create_model('DynamicLatentSchema', **fields)


def latent_from_schema(
  schema: Dict[str, Any], components: list[ComponentConfig]
) -> np.ndarray:
  """
  • Boolean mode  (one component, many flags)
  • Slot mode     (>=2 components, each one categorical)
  • Mixed mode    (components with .components inside - not used here)
  """
  boolean_mode = len(components) == 1 and not components[0].is_multi_component()
  size = sum(c.get_num_elements() for c in components)
  latent = np.zeros(size, dtype=int)
  idx = 0

  # ---------- 1) simple Boolean flags -----------------------------------
  if boolean_mode:
    comp = components[0]
    for elem in comp.elements:
      key = elem.replace('one ', '').replace(' ', '_')
      if schema.get(key):
        latent[idx] = 1
      idx += 1
    return latent

  # ---------- 2) slot-per-component  ------------------------------------
  for comp in components:
    if not comp.is_multi_component():
      chosen = schema.get(comp.name)
      for j, elem in enumerate(comp.elements):
        if strip_article(elem).replace('one ', '') == strip_article(chosen):
          latent[idx + j] = 1
          break
      idx += comp.get_num_elements()

    else:
      for sub in comp.components:
        chosen = schema.get(sub.name)
        for j, elem in enumerate(sub.elements):
          if elem.replace('one ', '') == chosen:
            latent[idx + j] = 1
            break
        idx += sub.get_num_elements()

  return latent


def build_prompt(
  cfg: Union['ComponentConfig', List['ComponentConfig']],
) -> str:
  """
  Create a clean prompt from either
  • a single ComponentConfig  → boolean yes/no for each element
  • a list of ComponentConfig → categorical slot-filling task
  """
  # ---------- Boolean: one component, many elements -----------------------
  if isinstance(cfg, ComponentConfig) and not getattr(cfg, 'components', None):
    items = ', '.join(cfg.elements)
    return (
      'Look at the image and decide **for each** of the following items '
      f'whether it is present:\n{items}\n\n'
      'Return **only** a JSON object whose keys are those items and '
      'whose values are booleans (true / false).'
    )

  # ---------- Categorical (multi-slot) -----------------------------------
  cfg_list = cfg if isinstance(cfg, list) else cfg.components
  bullet_lines = [f'• **{c.name}** - one of: {", ".join(c.elements)}' for c in cfg_list]
  keys_snippet = '{ ' + ', '.join(f'"{c.name}": "..."' for c in cfg_list) + ' }'

  return (
    'Identify the correct value for each slot below:\n'
    + '\n'.join(bullet_lines)
    + '\n\nReturn a JSON object with **exactly** these keys:\n'
    + keys_snippet
  )


def create_prompt_for_components(comp_setup: str, components: List[Any]) -> str:
  """
  Return a ready-to-send prompt string for the current component setup.
  """
  cfg = COMPONENT_CONFIGS[comp_setup]

  if getattr(cfg, 'prompt_template', None):
    return cfg.prompt_template

  if cfg.is_multi_component():
    return build_prompt(cfg.components)
  else:
    return build_prompt(cfg)


def load_and_correct_dataset(
  model_name: str, comp_setup: str, base_dir: Path
) -> pd.DataFrame:
  """
  Load dataset and correct file paths if necessary.

  Handles different path formats and corrects them to ensure all image
  paths are valid in the current execution environment. This is particularly
  useful when running evaluations in different environments.

  Args:
      model_name: Name of the model
      comp_setup: Name of the compositional setup
      base_dir: Base directory where data is stored

  Returns:
      DataFrame with corrected image paths

  Raises:
      FileNotFoundError: If the compositional setup directory doesn't exist
  """
  model_dir_name = model_name.replace('/', '_')
  comp_setup_dir = base_dir / model_dir_name / comp_setup

  if not comp_setup_dir.exists():
    raise FileNotFoundError(
      f'Compositional setup directory not found at {comp_setup_dir}'
    )

  logger.info(f'Using compositional setup directory: {comp_setup_dir}')

  dataset_path = comp_setup_dir / 'dataset_with_paths.pkl'
  logger.info(f'Loading dataset from {dataset_path}')
  dataset = pd.read_pickle(dataset_path)

  if 'image_path' in dataset.columns:
    valid_paths = 0
    for path in dataset['image_path']:
      if os.path.exists(path):
        valid_paths += 1

    logger.info(f'Initial check: {valid_paths}/{len(dataset)} paths exist')

    if valid_paths < len(dataset):
      for i, path in enumerate(dataset['image_path']):
        path_str = str(path)

        if 'compositional_experiment/' in path_str:
          rel_path = path_str.split('compositional_experiment/')[1]
          dataset.at[i, 'image_path'] = str(base_dir / rel_path)

        elif not os.path.exists(path_str):
          filename = os.path.basename(path_str)
          img_dir = comp_setup_dir / 'images'
          if os.path.exists(img_dir / filename):
            dataset.at[i, 'image_path'] = str(img_dir / filename)

      valid_paths_after = 0
      for path in dataset['image_path']:
        if os.path.exists(path):
          valid_paths_after += 1

      logger.info(f'After fixing: {valid_paths_after}/{len(dataset)} paths exist')

  return dataset


def evaluate_generalization(
  model_name: str,
  comp_setup: str,
  gemini_model_name: str = 'gemini-2.0-flash',
  base_dir: Optional[str] = None,
  metric_name: str = 'binary_accuracy',
  max_images: Optional[int] = None,
) -> Dict[str, Any]:
  """
  Main function to evaluate compositional generalization on a model.

  Orchestrates the full evaluation workflow:
  1. Locates experiment directory and loads dataset
  2. Sets up resumption capabilities via partial CSV results
  3. Selects test rows to evaluate
  4. Runs LLM analysis on images
  5. Computes and saves evaluation metrics

  Args:
      model_name: Name of the image generation model to evaluate
      comp_setup: Name of the compositional setup
      gemini_model_name: Name of the LLM model to use for evaluation
      base_dir: Base directory where experiment data is stored
      metric_name: Name of the metric to use for evaluation
      max_images: Maximum number of test images to evaluate

  Returns:
      Dictionary with results dataframe, summary metrics, and results directory
  """
  if base_dir is None:
    base_dir = Path(__file__).absolute().parent / 'compositional_experiment'
  else:
    base_dir = Path(base_dir).absolute()

  logger.info('Base dir: %s', base_dir)

  if not base_dir.exists():
    alt = Path.cwd() / 'compositional_experiment'
    if alt.exists():
      base_dir = alt
      logger.info('Using alternative base dir: %s', base_dir)

  dataset = load_and_correct_dataset(model_name, comp_setup, base_dir)

  model_dir_name = model_name.replace('/', '_')
  model_dir = base_dir / model_dir_name / comp_setup

  cfg = COMPONENT_CONFIGS[comp_setup]
  components = cfg.components if cfg.is_multi_component() else [cfg]
  schema_cls = create_schema_for_components(components)
  prompt_tmpl = create_prompt_for_components(comp_setup, components)

  out_dir = model_dir / f'gemini_eval_{gemini_model_name.replace("/", "_")}'
  out_dir.mkdir(parents=True, exist_ok=True)
  partial_csv = out_dir / 'partial_results.csv'

  done_paths: set[str] = set()
  prev_df: Optional[pd.DataFrame] = None
  if partial_csv.exists():
    prev_df = pd.read_csv(partial_csv)
    done_paths = set(prev_df['image_path'])
    logger.info('Resume: %d images already graded.', len(done_paths))

  test_images = dataset[dataset['split'] == 'test']
  if max_images and max_images < len(test_images):
    test_images = test_images.sample(max_images, random_state=42)

  test_images = test_images[~test_images['image_path'].isin(done_paths)]
  logger.info('Calling Gemini on %d new images …', len(test_images))

  paths = list(test_images['image_path'])
  ground_truth = list(test_images['latent'].values)

  if test_images.empty:
    logger.info('All test images have already been graded — assembling final report.')
    results = []
  else:
    analyzer = LatentAnalyzer(model_name=gemini_model_name)

    results = analyzer.analyze_images(
      image_paths=paths,
      schema_class=schema_cls,
      prompt_template=prompt_tmpl,
      ground_truth_latents=ground_truth,
      components=components,
      append_to_csv=partial_csv,
    )

  if results and all(r is None for r in results):
    raise RuntimeError('No successful Gemini analyses.')

  new_latents = [
    latent_from_schema(r, components)
    if r is not None
    else np.zeros(sum(c.get_num_elements() for c in components))
    for r in results
  ]
  new_df = pd.DataFrame(
    {
      'image_path': paths,
      'prediction': new_latents,
      'ground_truth': ground_truth,
      'prompt': test_images['prompt'].values,
    }
  )
  full_df = None
  if prev_df is not None:
    prev_df['prediction'] = prev_df['gemini_json'].apply(
      lambda s: latent_from_schema(json.loads(s), components)
    )
    lookup = dataset.set_index('image_path')
    prev_df['ground_truth'] = prev_df['image_path'].map(lookup['latent'])
    prev_df['prompt'] = prev_df['image_path'].map(lookup['prompt'])
    prev_df = prev_df.drop(columns='gemini_json')
    full_df = pd.concat([prev_df, new_df], ignore_index=True)

  else:
    full_df = new_df

  if test_images.empty and partial_csv.exists():
    partial_csv.unlink()

  predictions = np.stack(full_df['prediction'].values)
  ground_truth = np.stack(full_df['ground_truth'].values)

  from compositional_gen.metrics import get_metric

  metric_fn = get_metric(metric_name)
  metric_vals = metric_fn(predictions, ground_truth)

  binary = (predictions > 0.5).astype(int)
  acc = np.mean((binary == ground_truth).all(axis=1))
  per_c = np.mean((binary == ground_truth), axis=0)

  full_df[metric_name] = metric_vals
  full_df['fully_correct'] = (binary == ground_truth).all(axis=1)

  full_df.to_csv(out_dir / 'evaluation_results.csv', index=False)

  summary = dict(
    binary_accuracy=float(acc),
    per_component_accuracy=per_c.tolist(),
    fully_correct_count=int(full_df['fully_correct'].sum()),
    total_images=len(full_df),
    percent_correct=float(full_df['fully_correct'].mean() * 100),
  )
  with (out_dir / 'summary.json').open('w') as f:
    json.dump(summary, f, indent=2, cls=NumpyJSONEncoder)

  logger.info('Binary accuracy: %.4f', acc)
  logger.info(
    'Fully correct: %d/%d (%.2f%%)',
    summary['fully_correct_count'],
    summary['total_images'],
    summary['percent_correct'],
  )

  return {'results_df': full_df, 'summary': summary, 'results_dir': out_dir}


def _run_one_setup(
  *, model_name: str, comp_setup: str, args, base_dir: Optional[Path]
) -> int:
  logger.info('⧗  Evaluating  %s  /  %s', model_name, comp_setup)
  evaluate_generalization(
    model_name=model_name,
    comp_setup=comp_setup,
    gemini_model_name=args.gemini_model,
    base_dir=str(base_dir) if base_dir else None,
    metric_name=args.metric,
    max_images=args.max_images,
  )
  return 0


def _build_arg_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    description='Evaluate compositional generalisation with LLM-based analysis'
  )
  p.add_argument('--model-name', type=str, help='Diffusion model name')
  p.add_argument(
    '--model-choice', type=int, choices=range(len(AVAILABLE_MODELS)), default=5
  )
  p.add_argument('--base-dir', type=str, help='Base experiment directory')
  p.add_argument(
    '--comp-setup',
    type=str,
    choices=list(COMPONENT_CONFIGS.keys()),
    default='counting_animals',
  )
  p.add_argument('--gemini-model', type=str, default='gemini-2.0-flash')
  p.add_argument(
    '--metric',
    type=str,
    choices=['binary_accuracy', 'mse', 'mae', 'cosine'],
    default='binary_accuracy',
  )
  p.add_argument('--max-images', type=int)
  p.add_argument(
    '--all-setups',
    action='store_true',
    help='Evaluate **all** comp-setups found for the model',
    default=True,
  )
  return p


def main() -> int:
  parser = _build_arg_parser()
  args = parser.parse_args()

  model_name = args.model_name or AVAILABLE_MODELS[args.model_choice]
  gemini_tag = args.gemini_model.replace('/', '_')
  base_dir = Path(args.base_dir).expanduser() if args.base_dir else None

  if not args.all_setups:
    return _run_one_setup(
      model_name=model_name,
      comp_setup=args.comp_setup,
      args=args,
      base_dir=base_dir,
    )

  try:
    root = base_dir or Path(__file__).parent / 'compositional_experiment'
    model_dir = root / model_name.replace('/', '_')
    setups = find_all_comp_setups(model_dir)
    if not setups:
      logger.error('No compositional setups found in %s', model_dir)
      return 1

    todo = [s for s in setups if needs_eval(model_dir, s, gemini_tag)]
    logger.info('Found %d setups, %d still need evaluation', len(setups), len(todo))

    for comp_setup in todo:
      _run_one_setup(
        model_name=model_name,
        comp_setup=comp_setup,
        args=args,
        base_dir=base_dir,
      )
    return 0
  except Exception as exc:
    logger.error('Batch evaluation failed: %s', exc, exc_info=True)
    return 1


if __name__ == '__main__':
  exit(main())

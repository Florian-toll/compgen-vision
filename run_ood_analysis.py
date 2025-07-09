#!/usr/bin/env python3
"""
Rank OOD prompts for an *existing* experiment run.

All heavy lifting happens inside ``CompositionalGeneralizationAnalyzer``.
"""

from __future__ import annotations

import argparse
import logging
import os
import traceback
from pathlib import Path
from typing import Any
from typing import Dict

import pandas as pd
from compositional_gen.analysis import CompositionalGeneralizationAnalyzer
from compositional_gen.config import AVAILABLE_MODELS
from compositional_gen.config import COMPONENT_CONFIGS
from compositional_gen.data import CompositePromptGenerator
from compositional_gen.utils import get_available_layers

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s  %(levelname)-8s  %(message)s',
)
logger = logging.getLogger(__name__)


def _dataset_path(base: Path, model_name: str, setup: str) -> Path:
  return base / model_name.replace('/', '_') / setup / 'dataset_with_paths.pkl'


def _load_dataset(base: Path, model_name: str, setup: str) -> pd.DataFrame:
  pkl = _dataset_path(base, model_name, setup)
  if not pkl.exists():
    raise FileNotFoundError(pkl)
  df = pd.read_pickle(pkl)

  for col in ('activation_path', 'image_path'):
    if col not in df:
      continue
    df[col] = df[col].apply(
      lambda s: str(base / s.split('compositional_experiment/')[-1])
      if 'compositional_experiment/' in s
      else s
    )
  return df


def resolve_experiment_dir(arg_base_dir: str | None) -> Path:
  if arg_base_dir:
    return Path(arg_base_dir).expanduser()

  scratch = os.environ.get('SCRATCH')
  if scratch:
    return Path(scratch) / 'compositional_experiment'

  HERE = Path(__file__).resolve().parent
  candidate = HERE / 'compositional_experiment'
  if candidate.exists():
    return candidate

  return Path.home() / 'scratch' / 'compositional_experiment'


def run_analysis(
  model_name: str,
  comp_setup: str,
  base_dir: Path,
  *,
  all_layers: bool,
  only_layer: str | None,
  metric: str,
  write_img: bool,
  force_recompute: bool,
) -> Dict[str, Any]:
  ana = CompositionalGeneralizationAnalyzer(str(base_dir), compositional_setup=comp_setup)
  ana.current_model_name = model_name
  ana.dataset = _load_dataset(base_dir, model_name, comp_setup)

  cfg = COMPONENT_CONFIGS[comp_setup]
  ana.prompt_generator = (
    CompositePromptGenerator(cfg.components, template=cfg.template)
    if cfg.is_multi_component()
    else CompositePromptGenerator([cfg], hotness=3)
  )

  dec = ana.train_decoder_with_split(
    layer_name=only_layer,
    use_predefined_split=True,
    force_recompute=force_recompute,
  )

  if only_layer:
    ranked_df = ana.rank_ood_samples(
      dec,
      metric_name=metric,
      write_images=write_img,
      clear_existing=False,
    )
    return dict(layer=only_layer, ranking_df=ranked_df, decoder_results=dec)

  if all_layers:
    ranked = ana.rank_layers(dec, metric_name=metric, write_images=write_img)
    return dict(ranked=ranked, decoder_results=dec)

  best_layer = max(dec, key=lambda k: dec[k]['test_f1_score'])
  ranked_df = ana.rank_ood_samples(
    dec, metric_name=metric, write_images=write_img, clear_existing=False
  )
  return dict(best_layer=best_layer, ranking_df=ranked_df, decoder_results=dec)


def _arg_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description='Layer-wise OOD ranking on an existing run')

  p.add_argument('--model-name', type=str, help='HuggingFace model name')
  p.add_argument(
    '--model-choice',
    type=int,
    default=5,
    choices=range(len(AVAILABLE_MODELS)),
    help='Index into AVAILABLE_MODELS (default 3 - SD-3.5-medium)',
  )
  p.add_argument(
    '--comp-setup',
    type=str,
    default='animals_long',
    choices=list(COMPONENT_CONFIGS.keys()),
    help='Compositional setup to use (default fruits_veggies_long)',
  )
  p.add_argument(
    '--base-dir',
    type=str,
    help='Experiment root folder (default ./compositional_experiment)',
  )

  p.add_argument(
    '--all-layers',
    action='store_true',
    default=True,
    help='Rank every layer (default)',
  )
  p.add_argument(
    '--layer',
    type=str,
    help='Rank ONE specific layer (overrides --all-layers / best-layer mode)',
  )
  p.add_argument(
    '--force-recompute',
    action='store_true',
    default=True,
    help='Re-train decoder(s) even if cached (default)',
  )
  p.add_argument(
    '--metric',
    type=str,
    default='logloss',
    choices=['logloss', 'mse', 'mae', 'cosine', 'binary_accuracy'],
    help='Metric to use for ranking (default logloss)',
  )
  p.add_argument(
    '--skip-images',
    action='store_true',
    default=True,
    help='Do not copy ranked PNGs (only CSVs) — default on',
  )

  p.add_argument(
    '--list-layers',
    action='store_true',
    help='Print available layers in this run and exit',
  )

  return p


def main() -> int:
  args = _arg_parser().parse_args()

  model_name = args.model_name or AVAILABLE_MODELS[args.model_choice]
  base_dir = resolve_experiment_dir(args.base_dir)

  if args.list_layers:
    layers = get_available_layers(
      base_dir / model_name.replace('/', '_') / args.comp_setup
    )
    print('\n'.join(layers or ['<no layer info - run training first>']))
    return 0

  try:
    res = run_analysis(
      model_name,
      args.comp_setup,
      base_dir=base_dir,
      all_layers=args.all_layers and not args.layer,
      only_layer=args.layer,
      metric=args.metric,
      write_img=not args.skip_images,
      force_recompute=args.force_recompute,
    )
    if 'best_layer' in res:
      best = res['best_layer']
      logger.info(
        'Best layer: %s  (f1 %.4f)',
        best,
        res['decoder_results'][best]['test_f1_score'],
      )
    return 0
  except Exception as exc:
    logger.error('ERROR: %s', exc)
    traceback.print_exc()
    return 1


if __name__ == '__main__':
  logging.getLogger('PIL').setLevel(logging.WARNING)
  exit(main())

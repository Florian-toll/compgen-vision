#!/usr/bin/env python3
"""
Main script for running compositional generalization experiments.
"""

import argparse
import logging
import os
import shutil
from pathlib import Path

from compositional_gen.analysis import CompositionalGeneralizationAnalyzer
from compositional_gen.config import AVAILABLE_MODELS
from compositional_gen.config import COMPONENT_CONFIGS
from compositional_gen.config import AnalysisConfig
from compositional_gen.data import CompositePromptGenerator
from compositional_gen.data import create_all_combinations_dataset
from compositional_gen.data import create_cross_component_dataset
from compositional_gen.utils import ensure_directory

import wandb

logging.basicConfig(
  level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
  """Parse command line arguments."""
  parser = argparse.ArgumentParser(
    description='Run compositional generalization experiments'
  )
  parser.add_argument(
    '--output-dir',
    type=str,
    default=None,
    help='Directory to save outputs',
  )
  parser.add_argument(
    '--model-choice',
    type=int,
    default=0,
    choices=range(len(AVAILABLE_MODELS)),
    help='Model to use (0=SD1.5, 1=SDXL, 2=SD3-medium, 3=SD3.5-medium, 4=SD3.5-large, 5=FLUX.1-dev, 6=FLUX.1-schnell)',
  )
  parser.add_argument(
    '--model-name',
    type=str,
    help='Model name (overrides model-choice if provided)',
  )
  parser.add_argument(
    '--comp-setup',
    type=str,
    default='animals_short',
    choices=list(COMPONENT_CONFIGS.keys()),
    help='Compositional setup to use',
  )
  parser.add_argument(
    '--max-hotness',
    type=int,
    default=3,
    help='Maximum number of hot elements in each combination',
  )
  parser.add_argument(
    '--variable-hotness',
    action='store_true',
    help='Include combinations with 1 to max-hotness elements',
  )
  parser.add_argument(
    '--max-samples',
    type=int,
    default=None,
    help='Maximum number of samples to use (None for all combinations)',
  )
  parser.add_argument(
    '--hf-token',
    type=str,
    default=None,
    help='HuggingFace token for accessing gated models',
  )
  parser.add_argument(
    '--reuse-existing',
    action='store_true',
    help='Reuse existing images and activations if available',
  )
  parser.add_argument(
    '--metric',
    type=str,
    default='logloss',
    choices=['logloss', 'mse', 'mae', 'cosine', 'binary_accuracy'],
    help='Metric to use for ranking OOD samples',
  )
  parser.add_argument(
    '--num-seeds', type=int, default=1, help='How many random seeds to draw per prompt'
  )
  return parser.parse_args()


def get_output_dir(args_output_dir):
  """Determine the output directory."""
  if args_output_dir is not None:
    return Path(args_output_dir)

  scratch_dir = os.environ.get('SCRATCH')
  if scratch_dir:
    return Path(scratch_dir) / 'compositional_experiment'

  return Path.home() / 'scratch' / 'compositional_experiment'


def run_experiment(config: AnalysisConfig, reuse_existing: bool = False):
  """
  Run the compositional generalization experiment.

  Args:
      config: Configuration object
      reuse_existing: Whether to reuse existing data if available
  """
  logger.info(f'Starting experiment with model: {config.model_name}')
  logger.info(f'Compositional setup: {config.compositional_setup}')
  logger.info(f'Output directory: {config.output_dir}')

  analyzer = CompositionalGeneralizationAnalyzer(
    output_dir=str(config.output_dir),
    cache_dir=str(config.cache_dir) if config.cache_dir else None,
    compositional_setup=config.compositional_setup,
  )

  selected_config = COMPONENT_CONFIGS[config.compositional_setup]

  model_dir = analyzer.get_model_dir(config.model_name)
  ensure_directory(model_dir)

  if selected_config.is_multi_component():
    if not reuse_existing or not (model_dir / 'dataset_with_paths.pkl').exists():
      dataset = create_cross_component_dataset(
        components=selected_config.components,
        template=selected_config.template,
        test_size=0.3,
        random_state=42,
        max_samples=config.max_samples,
      )
      analyzer.dataset = dataset

      analyzer.prompt_generator = CompositePromptGenerator(
        components=selected_config.components, template=selected_config.template
      )
      logger.info(f'Created cross-component dataset with {len(dataset)} samples')
  else:
    if not reuse_existing or not (model_dir / 'dataset_with_paths.pkl').exists():
      dataset = create_all_combinations_dataset(
        components=[selected_config],
        hotness=config.hotness,
        test_size=0.3,
        random_state=42,
        max_samples=config.max_samples,
        variable_hotness=config.variable_hotness,
      )
      analyzer.dataset = dataset

      analyzer.prompt_generator = CompositePromptGenerator(
        components=[selected_config], hotness=config.hotness
      )
      logger.info(f'Created traditional dataset with {len(dataset)} samples')

  analyzer.generate_images(
    model_name=config.model_name,
    num_inference_steps=30,
    num_seeds=config.num_seeds,
    reuse_existing=reuse_existing,
  )

  logger.info('Training decoder using predefined train/test split...')
  decoder_results = analyzer.train_decoder_with_split(
    use_predefined_split=True,
    force_recompute=not reuse_existing,
  )

  logger.info('Evaluating generalization...')
  analyzer.evaluate_generalization(decoder_results)
  analyzer.rank_layers(
    decoder_results,
    metric_name=config.metric,
    write_images=True,
  )

  logger.info(f'Ranking OOD samples by {config.metric}...')
  analyzer.rank_ood_samples(
    decoder_results,
    metric_name=config.metric,
  )
  wandb.finish()

  logger.info(f'Results saved to {model_dir}')

  zip_path = model_dir.parent / f'{config.compositional_setup}.zip'
  logger.info(f'Zipping experiment folder to {zip_path}...')
  try:
    shutil.make_archive(
      base_name=str(zip_path).replace('.zip', ''),
      format='zip',
      root_dir=model_dir.parent,
      base_dir=model_dir.name,
    )
    logger.info(f'Experiment zipped to {zip_path}')
  except Exception as e:
    logger.error(f'Error zipping experiment folder: {e}')

  return 0


def main():
  """Main function."""
  args = parse_args()

  settings = wandb.Settings(
    _service_wait=10,
  )

  wandb.init(
    project='image_composition',
    config=vars(args),
    settings=settings,
    dir='/tmp/wandb_logs',
  )

  output_dir = get_output_dir(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  model_name = args.model_name
  if model_name is None:
    model_name = AVAILABLE_MODELS[args.model_choice]

  if args.hf_token:
    os.environ['HUGGINGFACE_TOKEN'] = args.hf_token

  config = AnalysisConfig(
    model_name=model_name,
    compositional_setup=args.comp_setup,
    output_dir=output_dir,
    hotness=args.max_hotness,
    variable_hotness=args.variable_hotness,
    max_samples=args.max_samples,
    metric=args.metric,
    num_seeds=args.num_seeds,
  )

  try:
    run_experiment(config, reuse_existing=args.reuse_existing)
  except Exception as e:
    logger.error(f'Error occurred: {e}', exc_info=True)
    return 1


if __name__ == '__main__':
  main()

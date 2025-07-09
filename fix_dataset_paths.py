#!/usr/bin/env python3
"""
Script to fix dataset paths in dataset_with_paths.pkl files to match the new directory structure.
This updates all model directories and compositional setups at once, fixing broken absolute paths.
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(
  level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fix_paths(base_dir: Path, dry_run: bool = False) -> None:
  """
  Fix paths in all dataset_with_paths.pkl files found under the base directory.

  Args:
      base_dir: Base directory where model directories are stored
      dry_run: If True, only report what would be changed without actually making changes
  """
  total_datasets = 0
  total_activation_paths_fixed = 0
  total_image_paths_fixed = 0

  model_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
  logger.info(f'Found {len(model_dirs)} model directories')

  base_dir_name = base_dir.name

  for model_dir in model_dirs:
    model_name = model_dir.name
    logger.info(f'Processing model directory: {model_name}')

    comp_dirs = [d for d in model_dir.iterdir() if d.is_dir()]

    for comp_dir in comp_dirs:
      comp_setup = comp_dir.name
      logger.info(f'  Processing compositional setup: {comp_setup}')

      dataset_path = comp_dir / 'dataset_with_paths.pkl'
      if not dataset_path.exists():
        logger.warning(f'  No dataset file found at {dataset_path}, skipping')
        continue

      total_datasets += 1

      try:
        logger.info(f'  Loading dataset from {dataset_path}')
        dataset = pd.read_pickle(dataset_path)

        needs_fixing = False

        if 'activation_path' in dataset.columns and len(dataset) > 0:
          sample_path = dataset['activation_path'].iloc[0]
          if sample_path:
            if os.path.isabs(sample_path):
              needs_fixing = True
              logger.info(
                f"  Dataset needs fixing: path '{sample_path}' needs correction"
              )

        if not needs_fixing:
          logger.info('  Paths appear to be correct, skipping')
          continue

        activation_paths_fixed = 0
        image_paths_fixed = 0

        sample_count = 0
        max_samples = 3

        if 'activation_path' in dataset.columns:
          for i, path in enumerate(dataset['activation_path']):
            if path:
              path_obj = Path(path)
              filename = path_obj.name

              corrected_path = str(
                Path(base_dir_name) / model_name / comp_setup / 'activations' / filename
              )

              if sample_count < max_samples:
                logger.info(f'  Sample path transformation {sample_count+1}:')
                logger.info(f'    Original: {path}')
                logger.info(f'    New: {corrected_path}')
                sample_count += 1

              if not dry_run:
                dataset.at[i, 'activation_path'] = corrected_path

              activation_paths_fixed += 1

        if 'image_path' in dataset.columns:
          for i, path in enumerate(dataset['image_path']):
            if path:
              path_obj = Path(path)
              filename = path_obj.name

              corrected_path = str(
                Path(base_dir_name) / model_name / comp_setup / 'images' / filename
              )

              if sample_count < max_samples:
                logger.info(f'  Sample path transformation {sample_count+1}:')
                logger.info(f'    Original: {path}')
                logger.info(f'    New: {corrected_path}')
                sample_count += 1

              if not dry_run:
                dataset.at[i, 'image_path'] = corrected_path

              image_paths_fixed += 1

        if not dry_run and (activation_paths_fixed > 0 or image_paths_fixed > 0):
          backup_path = dataset_path.with_suffix('.pkl.backup')
          if not backup_path.exists():
            import shutil

            shutil.copy2(dataset_path, backup_path)
            logger.info(f'  Created backup at {backup_path}')

          dataset.to_pickle(dataset_path)
          logger.info(f'  Saved updated dataset to {dataset_path}')

        logger.info(
          f'  Fixed {activation_paths_fixed} activation paths and {image_paths_fixed} image paths'
        )

        total_activation_paths_fixed += activation_paths_fixed
        total_image_paths_fixed += image_paths_fixed

      except Exception as e:
        logger.error(f'  Error processing {dataset_path}: {e}')

  action = 'Would fix' if dry_run else 'Fixed'
  logger.info(f'Summary: Processed {total_datasets} datasets')
  logger.info(
    f'{action} {total_activation_paths_fixed} activation paths and {total_image_paths_fixed} image paths'
  )


def main():
  """Main function."""
  parser = argparse.ArgumentParser(
    description='Fix paths in dataset_with_paths.pkl files to match the new directory structure'
  )
  parser.add_argument(
    '--base-dir',
    type=str,
    default=None,
    help='Base directory where model directories are stored (defaults to ./compositional_experiment)',
  )
  parser.add_argument(
    '--dry-run',
    action='store_true',
    help='Only report what would be changed without actually making changes',
  )
  args = parser.parse_args()

  if args.base_dir is None:
    script_dir = Path(__file__).absolute().parent
    base_dir = script_dir / 'compositional_experiment'
  else:
    base_dir = Path(args.base_dir).absolute()

  if not base_dir.exists():
    logger.error(f'Base directory {base_dir} does not exist!')
    return 1

  logger.info(f'Using base directory: {base_dir}')
  logger.info(f'Dry run: {args.dry_run}')

  fix_paths(base_dir, dry_run=args.dry_run)

  return 0


if __name__ == '__main__':
  exit_code = main()
  exit(exit_code)

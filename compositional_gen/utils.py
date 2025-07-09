"""
Utility functions for the compositional generalization analysis.

This module provides utility functions for file operations, directory management,
and other common tasks.
"""

import json
import logging
import pickle
import shutil
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

logging.basicConfig(
  level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clear_directory(directory_path: Union[str, Path]) -> None:
  """
  Clear all contents of a directory without removing the directory itself.

  Args:
      directory_path: Path to the directory to clear
  """
  path = Path(directory_path)
  if path.exists():
    for item in path.iterdir():
      if item.is_dir():
        shutil.rmtree(item)
      else:
        item.unlink()
    logger.info(f'Cleared directory: {directory_path}')
  else:
    logger.info(f"Directory doesn't exist, will be created: {directory_path}")
    path.mkdir(parents=True, exist_ok=True)


def ensure_directory(directory_path: Union[str, Path]) -> Path:
  """
  Ensure a directory exists, creating it if necessary.

  Args:
      directory_path: Path to the directory

  Returns:
      Path object for the directory
  """
  path = Path(directory_path)
  path.mkdir(parents=True, exist_ok=True)
  return path


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
  """
  Save data to a JSON file.

  Args:
      data: Data to save
      file_path: Path to the output file
  """
  with open(file_path, 'w') as f:
    json.dump(data, f, indent=2)
  logger.debug(f'Saved JSON data to {file_path}')


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
  """
  Load data from a JSON file.

  Args:
      file_path: Path to the JSON file

  Returns:
      Loaded data

  Raises:
      FileNotFoundError: If the file does not exist
      json.JSONDecodeError: If the file is not valid JSON
  """
  with open(file_path, 'r') as f:
    data = json.load(f)
  logger.debug(f'Loaded JSON data from {file_path}')
  return data


def save_pickle(data: Any, file_path: Union[str, Path]) -> None:
  """
  Save data to a pickle file.

  Args:
      data: Data to save
      file_path: Path to the output file
  """
  with open(file_path, 'wb') as f:
    pickle.dump(data, f)
  logger.debug(f'Saved pickle data to {file_path}')


def load_pickle(file_path: Union[str, Path]) -> Any:
  """
  Load data from a pickle file.

  Args:
      file_path: Path to the pickle file

  Returns:
      Loaded data

  Raises:
      FileNotFoundError: If the file does not exist
  """
  with open(file_path, 'rb') as f:
    data = pickle.load(f)
  logger.debug(f'Loaded pickle data from {file_path}')
  return data


def get_available_layers(model_dir: Path) -> List[str]:
  """
  Get a list of available layers from decoder results.

  Args:
      model_dir: Path to the model directory

  Returns:
      List of layer names
  """
  cache_file = model_dir / 'cached_decoder_results.pkl'
  if cache_file.exists():
    with open(cache_file, 'rb') as f:
      cached_results = pickle.load(f)
    return list(cached_results.keys())

  results_file = model_dir / 'decoder_results.json'
  if results_file.exists():
    with open(results_file, 'r') as f:
      results = json.load(f)
    return list(results.keys())

  return []


def copy_ranked_images(
  source_path: Path, dest_dir: Path, rank: int, animal_names: Optional[List[str]] = None
) -> None:
  """
  Copy an image file to a destination directory with a new name based on rank and animals.

  Args:
      source_path: Path to the source image
      dest_dir: Destination directory
      rank: Rank number for the image
      animal_names: Optional list of animal names for the filename
  """
  try:
    if animal_names and len(animal_names) > 0:
      animals_str = '_'.join(animal_names)
      dest_path = dest_dir / f'{rank}_{animals_str}.png'
    else:
      dest_path = dest_dir / f'{rank}.png'

    shutil.copy2(source_path, dest_path)
  except Exception as e:
    logger.error(f'Error copying {source_path} to {dest_dir}: {e}')

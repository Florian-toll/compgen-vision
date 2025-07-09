"""
LLM-based analysis for latent space interpretation.

This module provides utilities to analyze images using LLMs for extracting
latent space information.
"""

import json
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import llm
from pydantic import BaseModel

logging.basicConfig(
  level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LatentAnalyzer:
  """Analyze images using LLM for latent space interpretation."""

  def __init__(self, model_name: str = 'gemini-2.5-pro-exp-03-25'):
    """
    Initialize the latent analyzer.

    Args:
        model_name: Name of the LLM to use
    """
    try:
      import llm

      self.model = llm.get_model(model_name)
      self._has_llm = True
    except ImportError:
      logger.warning("LLM package not found. Install with 'pip install llm'")
      self._has_llm = False
    except Exception as e:
      logger.warning(f'Error initializing LLM model: {e}')
      self._has_llm = False

  def analyze_images(
    self, image_paths: List[Union[str, Path]], schema_class: type[BaseModel]
  ) -> List[Dict[str, Any]]:
    """
    Analyze images using the provided schema.

    Args:
        image_paths: List of paths to images
        schema_class: Pydantic model class for the schema

    Returns:
        List of dictionaries with extracted latent information

    Raises:
        RuntimeError: If LLM package is not available
    """
    if not self._has_llm:
      raise RuntimeError('LLM package not available')

    results = []

    for path in image_paths:
      path_str = str(path)
      logger.info(f'Analyzing image: {path_str}')

      try:
        response = self.model.prompt(
          'Describe the contents of the image according to the provided schema.',
          attachments=[llm.Attachment(path=path_str)],
          schema=schema_class,
        )
        latent = json.loads(response.text())
        results.append(latent)
        logger.info(f'Successfully analyzed image: {path_str}')
      except Exception as e:
        logger.error(f'Error analyzing image {path_str}: {e}')
        results.append(None)

    return results

  def batch_analyze_directory(
    self,
    directory: Union[str, Path],
    schema_class: type[BaseModel],
    pattern: str = '*.png',
    max_images: Optional[int] = None,
  ) -> Dict[str, Dict[str, Any]]:
    """
    Analyze all images in a directory matching the pattern.

    Args:
        directory: Directory containing images
        schema_class: Pydantic model class for the schema
        pattern: Glob pattern for image files
        max_images: Maximum number of images to analyze

    Returns:
        Dictionary mapping image paths to latent information
    """
    if not self._has_llm:
      raise RuntimeError('LLM package not available')

    dir_path = Path(directory)
    image_paths = list(dir_path.glob(pattern))

    if max_images is not None and max_images < len(image_paths):
      logger.info(f'Limiting analysis to {max_images} images')
      image_paths = image_paths[:max_images]

    results = self.analyze_images(image_paths, schema_class)

    return {str(path): result for path, result in zip(image_paths, results, strict=False)}

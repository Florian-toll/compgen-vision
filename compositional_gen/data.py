# Copyright (c) 2025 Florian Redhardt
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Data module for compositional generalization analysis.

This module contains dataset creation and component configuration classes.
"""

import itertools
import logging
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
  level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ComponentConfig:
  """Configuration for compositional components."""

  name: str
  elements: List[str]
  template: Optional[str] = None
  components: Optional[List['ComponentConfig']] = None
  prompt_template: Optional[str] = None

  def get_element(self, idx: int) -> str:
    """Get element by index."""
    return self.elements[idx]

  def get_num_elements(self) -> int:
    """Get number of elements in this component."""
    return len(self.elements)

  def is_multi_component(self) -> bool:
    """Check if this is a multi-component configuration."""
    return self.components is not None and len(self.components) > 0


class CompositePromptGenerator:
  """Generate compositional prompts from latent vectors."""

  def __init__(
    self,
    components: List[ComponentConfig],
    hotness: Optional[int] = None,
    template: Optional[str] = None,
  ):
    """Initialize prompt generator with components."""
    self.components = components
    self.num_components = len(components)
    self.hotness = hotness
    self.template = template

    self.component_indices = {}
    start_idx = 0
    for comp in components:
      num_elements = comp.get_num_elements()
      self.component_indices[comp.name] = (start_idx, start_idx + num_elements)
      start_idx += num_elements

    self.latent_dim = start_idx

  def get_components_from_latent(self, latent: np.ndarray) -> Dict[str, List[str]]:
    """
    Get active components and their elements from a latent vector.

    Args:
        latent: One-hot encoded latent vector

    Returns:
        Dictionary mapping component names to their active elements
    """
    result = {}
    for comp in self.components:
      start_idx, end_idx = self.component_indices[comp.name]
      comp_latent = latent[start_idx:end_idx]
      active_elements = np.where(comp_latent > 0)[0]

      if len(active_elements) > 0:
        result[comp.name] = [comp.get_element(idx) for idx in active_elements]

    return result


def create_cross_component_dataset(
  components: List[ComponentConfig],
  template: str,
  test_size: float = 0.3,
  random_state: int = 42,
  max_samples: Optional[int] = None,
) -> pd.DataFrame:
  """
  Create a dataset with all combinations of selecting one element from each
  component, plus a small grammar patch for the counting-objects template.
  """

  _IRREG = {
    'wolves': 'wolf',
    'tomatoes': 'tomato',
    'potatoes': 'potato',
    'foxes': 'fox',
  }

  def _singular(word: str) -> str:
    if word in _IRREG:
      return _IRREG[word]
    return word[:-1] if word.endswith('s') else word

  logger.info(f'Creating cross-component dataset with {len(components)} components')

  component_elements = {comp.name: comp.elements for comp in components}
  component_names = list(component_elements.keys())

  index_lists = [list(range(len(lst))) for lst in component_elements.values()]
  all_combinations = list(itertools.product(*index_lists))
  logger.info('Generated %d possible combinations', len(all_combinations))

  if max_samples and max_samples < len(all_combinations):
    np.random.seed(random_state)
    sel = np.random.choice(len(all_combinations), max_samples, replace=False)
    all_combinations = [all_combinations[i] for i in sel]
    logger.info('Limited to %d random samples', max_samples)

  latents, prompts = [], []

  latent_dim = sum(len(v) for v in component_elements.values())

  for combo in all_combinations:
    latent = np.zeros(latent_dim)
    prompt_dict = {}

    start = 0
    for comp_name, idx in zip(component_names, combo, strict=False):
      elems = component_elements[comp_name]
      latent[start + idx] = 1
      prompt_dict[comp_name] = elems[idx]
      start += len(elems)

    if prompt_dict.get('number') == 'one' and 'object' in prompt_dict:
      prompt_dict['object'] = _singular(prompt_dict['object'])

    prompts.append(template.format(**prompt_dict))
    latents.append(latent)

  df = pd.DataFrame({'prompt': prompts, 'latent': latents})

  train_idx, test_idx = train_test_split(
    df.index, test_size=test_size, random_state=random_state
  )
  df['split'] = 'train'
  df.loc[test_idx, 'split'] = 'test'

  logger.info(
    'Split dataset into %d training and %d testing samples', len(train_idx), len(test_idx)
  )
  return df


def create_all_combinations_dataset(
  components: List[ComponentConfig],
  hotness: int = 3,
  test_size: float = 0.3,
  random_state: int = 42,
  max_samples: Optional[int] = None,
  variable_hotness: bool = False,
) -> pd.DataFrame:
  """
  Create a dataset with k-hot combinations, split into train and test sets.

  Args:
      components: List of component configurations
      hotness: Maximum number of elements to activate
      test_size: Fraction of data to use for testing
      random_state: Random seed
      max_samples: Maximum number of samples (if None, use all combinations)
      variable_hotness: If True, include combinations with 1 to k hot elements

  Returns:
      DataFrame with prompts, latent vectors, and train/test split
  """
  logger.info(f'Creating dataset with {len(components)} components')
  component = components[0]
  elements = component.elements
  num_elements = len(elements)

  # Generate combinations based on variable_hotness setting
  all_combinations = []
  if variable_hotness:
    # Generate combinations with 1 to k hot elements
    for h in range(1, hotness + 1):
      combinations_for_h = list(itertools.combinations(range(num_elements), h))
      all_combinations.extend(combinations_for_h)
    logger.info(
      f'Generated all possible 1 to {hotness}-hot combinations: {len(all_combinations)} total combinations'
    )
  else:
    # Generate only k-hot combinations
    all_combinations = list(itertools.combinations(range(num_elements), hotness))
    logger.info(
      f'Generated all possible {hotness}-hot combinations: {len(all_combinations)} total combinations'
    )

  if max_samples is not None and max_samples < len(all_combinations):
    np.random.seed(random_state)
    selected_indices = np.random.choice(len(all_combinations), max_samples, replace=False)
    all_combinations = [all_combinations[i] for i in selected_indices]
    logger.info(f'Limited to {max_samples} random samples for faster testing')

  latents = []
  prompts = []
  hotness_values = []

  for combo in all_combinations:
    latent = np.zeros(num_elements)
    latent[list(combo)] = 1
    current_hotness = len(combo)

    selected_elements = [elements[idx] for idx in combo]
    prompt = 'An image with ' + ', '.join(f'one {e}' for e in selected_elements)

    latents.append(latent)
    prompts.append(prompt)
    hotness_values.append(current_hotness)

  dataset = pd.DataFrame(
    {'prompt': prompts, 'latent': latents, 'hotness': hotness_values}
  )

  train_indices, test_indices = train_test_split(
    range(len(dataset)),
    test_size=test_size,
    random_state=random_state,
    stratify=dataset['hotness'] if variable_hotness else None,
  )

  dataset['split'] = 'train'
  dataset.loc[test_indices, 'split'] = 'test'

  train_count = len(train_indices)
  test_count = len(test_indices)
  logger.info(
    f'Split dataset into {train_count} training and {test_count} testing samples'
  )

  if variable_hotness:
    logger.info('\nSample distribution by hotness:')
    for h in range(1, hotness + 1):
      count = (dataset['hotness'] == h).sum()
      train_count = ((dataset['hotness'] == h) & (dataset['split'] == 'train')).sum()
      test_count = ((dataset['hotness'] == h) & (dataset['split'] == 'test')).sum()
      logger.info(f'  {h}-hot: {count} total, {train_count} train, {test_count} test')

  return dataset

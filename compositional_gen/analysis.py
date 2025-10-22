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
Analysis module for compositional generalization experiments.

This module contains the main analyzer class and functions for OOD analysis.
"""

import logging
import os
import shutil
import warnings
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .metrics import get_metric
from .models import DiffusionModelWrapper
from .utils import clear_directory
from .utils import load_pickle
from .utils import save_json
from .utils import save_pickle

logging.basicConfig(
  level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=ConvergenceWarning)


class CompositionalGeneralizationAnalyzer:
  """Main class for analyzing compositional generalization in diffusion models."""

  def __init__(
    self,
    output_dir: str = 'compositional_experiment',
    cache_dir: Optional[str] = None,
    compositional_setup: str = 'animals_long',
  ):
    """
    Initialize the analyzer.

    Args:
        output_dir: Directory to save outputs
        cache_dir: Directory to cache models
        compositional_setup: Name of the compositional setup to use
    """
    self.output_dir = Path(output_dir)
    self.output_dir.mkdir(exist_ok=True, parents=True)

    self.cache_dir = cache_dir
    self.dataset = None
    self.model_wrapper = None
    self.prompt_generator = None
    self.current_model_name = None
    self.compositional_setup = compositional_setup

  def load_model(self, model_name: str) -> None:
    """
    Load a diffusion model.

    Args:
        model_name: HuggingFace model name
    """
    token = os.environ.get('HUGGINGFACE_TOKEN')
    cache_dir = self.cache_dir
    if cache_dir is None:
      cache_dir = os.environ.get('HF_HOME') or os.environ.get('TRANSFORMERS_CACHE')

    self.model_wrapper = DiffusionModelWrapper(
      model_name=model_name, cache_dir=cache_dir, token=token
    )
    self.current_model_name = model_name

  def generate_images(
    self,
    model_name: Optional[str] = None,
    num_inference_steps: int = 30,
    base_seed: int = 42,
    num_seeds: int = 1,
    clear_existing: bool = True,
    reuse_existing: bool = False,
  ) -> None:
    if self.dataset is None:
      raise ValueError('Dataset not loaded.')

    if model_name:
      self.load_model(model_name)
    elif self.model_wrapper is None:
      raise ValueError('No model loaded.')

    model_dir = self.get_model_dir()
    images_dir = model_dir / 'images'
    activ_dir = model_dir / 'activations'

    if reuse_existing and images_dir.exists() and activ_dir.exists():
      self.dataset = pd.read_pickle(model_dir / 'dataset_with_paths.pkl')
      logger.info('Re-using existing images/activations.')
      return

    if clear_existing:
      clear_directory(images_dir)
      clear_directory(activ_dir)
    else:
      images_dir.mkdir(parents=True, exist_ok=True)
      activ_dir.mkdir(parents=True, exist_ok=True)

    expanded_rows = []

    logger.info(
      'Generating %dx%d = %d images …',
      len(self.dataset),
      num_seeds,
      len(self.dataset) * num_seeds,
    )

    for idx, row in tqdm(self.dataset.iterrows(), total=len(self.dataset)):
      prompt = row['prompt']
      latent = row['latent']
      split = row['split']

      for k in range(num_seeds):
        seed = base_seed + idx * num_seeds + k

        img = self.model_wrapper.generate_image(
          prompt=prompt, num_inference_steps=num_inference_steps, seed=seed
        )

        img_path = images_dir / f'image_{idx:04d}_s{k}.png'
        img.save(img_path)

        acts = self.model_wrapper.get_activations(agg='cat+allmean')
        act_path = activ_dir / f'activ_{idx:04d}_s{k}.npz'
        np.savez_compressed(act_path, **acts)

        # clear UNet hook buffers to save RAM
        for hook in self.model_wrapper.hooks.values():
          hook._buf.clear()

        expanded_rows.append(
          {
            'prompt': prompt,
            'latent': latent,
            'split': split,
            'seed': k,
            'image_path': str(img_path),
            'activation_path': str(act_path),
          }
        )
    self.dataset = pd.DataFrame(expanded_rows).reset_index(drop=True)
    self.dataset.to_pickle(model_dir / 'dataset_with_paths.pkl')
    logger.info('Saved expanded dataset with %d rows.', len(self.dataset))

  def get_model_dir(self, model_name=None) -> Path:
    """
    Get the directory for the current model and compositional setup.

    Args:
        model_name: Optional model name to override current_model_name

    Returns:
        Path to model directory with compositional setup subdirectory
    """
    name_to_use = model_name if model_name is not None else self.current_model_name

    if name_to_use is None:
      raise ValueError('Model name not provided and current_model_name not set')

    model_dir = self.output_dir / name_to_use.replace('/', '_')
    return model_dir / self.compositional_setup

  def train_decoder_with_split(
    self,
    layer_name: Optional[str] = None,
    max_iter: int = 100,
    use_predefined_split: bool = True,
    force_recompute: bool = False,
  ) -> Dict[str, Any]:
    """
    Train a Logistic Regression decoder respecting the predefined train/test split.
    Reports both in-distribution (train) and out-of-distribution (test) performance
    using F1 score and log loss metrics.

    Args:
        layer_name: Specific layer to use (if None, will try all layers)
        max_iter: Maximum number of iterations for LogisticRegression
        use_predefined_split: Whether to use the predefined split column
        force_recompute: Whether to force recomputation even if cached results exist

    Returns:
        Dictionary with results
    """
    model_dir = self.get_model_dir()
    cache_file = model_dir / 'cached_decoder_results.pkl'

    if not force_recompute and cache_file.exists():
      try:
        cached_results = self.load_decoders(cache_file)

        if layer_name is not None and layer_name in cached_results:
          logger.info(f'Using cached decoder for layer: {layer_name}')
          return {layer_name: cached_results[layer_name]}
        elif layer_name is None:
          logger.info('Using cached decoders for all layers')
          return cached_results
      except Exception as e:
        logger.warning(f'Error loading cached decoders: {e}. Will recompute.')

    if self.dataset is None or 'activation_path' not in self.dataset.columns:
      raise ValueError('No dataset with activations available.')

    results = {}

    activation_paths = self.dataset['activation_path'].values
    latents = np.stack(self.dataset['latent'].values)

    first_activations = dict(np.load(activation_paths[0]))
    available_layers = list(first_activations.keys())

    layers_to_process = [layer_name] if layer_name else available_layers

    train_indices, test_indices = self._get_train_test_indices(use_predefined_split)

    for layer in layers_to_process:
      if layer not in available_layers:
        logger.warning(f'Layer {layer} not found in activations.')
        continue

      logger.info(f'Training decoder for layer {layer}…')

      # load activations
      X = [dict(np.load(p))[layer].astype(np.float32).flatten() for p in activation_paths]
      X = np.stack(X, dtype=np.float32)

      X_train, X_test = X[train_indices], X[test_indices]
      y_train, y_test = latents[train_indices], latents[test_indices]

      # fit
      decoder = make_pipeline(
        StandardScaler(with_mean=False),
        MultiOutputClassifier(
          LogisticRegression(
            max_iter=max_iter,
            n_jobs=-1,
            solver='lbfgs',
          )
        ),
      )
      decoder.fit(X_train, y_train)

      # SOFT probabilities
      proba_train = np.column_stack([p[:, 1] for p in decoder.predict_proba(X_train)])
      proba_test = np.column_stack([p[:, 1] for p in decoder.predict_proba(X_test)])

      # HARD bits (0/1)
      pred_bits_train = (proba_train > 0.5).astype(int)
      pred_bits_test = (proba_test > 0.5).astype(int)

      train_acc = (pred_bits_train == y_train).all(axis=1).mean()
      test_acc = (pred_bits_test == y_test).all(axis=1).mean()
      train_f1 = f1_score(y_train, pred_bits_train, average='weighted')
      test_f1 = f1_score(y_test, pred_bits_test, average='weighted')
      f1_gap = train_f1 - test_f1
      acc_gap = train_acc - test_acc

      results[layer] = {
        'model': decoder,
        'layer': layer,
        'train_f1_score': train_f1,
        'train_accuracy': train_acc,
        'test_f1_score': test_f1,
        'test_accuracy': test_acc,
        'f1_gap': f1_gap,
        'accuracy_gap': acc_gap,
        'probabilities_test': proba_test,
        'pred_bits_test': pred_bits_test,
        'ground_truth': y_test,
        'test_indices': test_indices,
      }

      logger.info(f'Layer {layer}:  Train F1 {train_f1:.4f}  Test F1 {test_f1:.4f}')

      logger.info(f'Layer {layer}:')
      logger.info(
        f'  In-distribution (Train) - F1: {train_f1:.4f}, Accuracy: {train_acc:.4f}'
      )
      logger.info(
        f'  Out-of-distribution (Test) - F1: {test_f1:.4f}, Accuracy: {test_acc:.4f}'
      )
      logger.info(f'  Generalization Gap - F1: {f1_gap:.4f}, Accuracy: {acc_gap:.4f}')

    model_dir = self.get_model_dir()
    results_file = model_dir / 'decoder_results.json'

    serializable_results = {}
    for layer, layer_results in results.items():
      serializable_results[layer] = {
        'train_f1_score': float(layer_results['train_f1_score']),
        'train_accuracy': float(layer_results['train_accuracy']),
        'test_f1_score': float(layer_results['test_f1_score']),
        'test_accuracy': float(layer_results['test_accuracy']),
        'f1_gap': float(layer_results['f1_gap']),
        'accuracy_gap': float(layer_results['accuracy_gap']),
        'layer': layer,
      }

    save_json(serializable_results, results_file)

    cache_file = model_dir / 'cached_decoder_results.pkl'
    save_pickle(results, cache_file)

    best_train_layer = max(results.keys(), key=lambda k: results[k]['train_f1_score'])
    best_test_layer = max(results.keys(), key=lambda k: results[k]['test_f1_score'])

    logger.info('\nBest Layers Summary:')
    logger.info(
      f"Best in-distribution layer: {best_train_layer} "
      f"(Train F1: {results[best_train_layer]['train_f1_score']:.4f})"
    )

    logger.info(
      f"Best out-of-distribution layer: {best_test_layer} "
      f"(Test F1: {results[best_test_layer]['test_f1_score']:.4f})"
    )

    self.save_decoders(results, cache_file)

    return results

  def save_decoders(
    self, decoder_results: Dict[str, Any], file_path: Optional[str] = None
  ) -> None:
    """
    Save decoder models and their results to a file.

    Args:
        decoder_results: Dictionary with decoder results to save
        file_path: Optional path to save file
        (defaults to cached_decoder_results.pkl in model dir)
    """
    if file_path is None:
      model_dir = self.get_model_dir()
      file_path = model_dir / 'cached_decoder_results.pkl'

    logger.info(f'Saving decoder results to {file_path}')
    save_pickle(decoder_results, file_path)

  def load_decoders(self, file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load decoder models and their results from a file.

    Args:
        file_path: Optional path to load file
        (defaults to cached_decoder_results.pkl in model dir)

    Returns:
        Dictionary with decoder results

    Raises:
        FileNotFoundError: If the file does not exist
    """
    if file_path is None:
      model_dir = self.get_model_dir()
      file_path = model_dir / 'cached_decoder_results.pkl'

    if not Path(file_path).exists():
      raise FileNotFoundError(f'Decoder file not found at {file_path}')

    logger.info(f'Loading decoder results from {file_path}')
    return load_pickle(file_path)

  # ----------------------------------------------------------
  def _get_train_test_indices(
    self, use_predefined_split: bool
  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Return *positional* indices (0 … N-1) for train and test.
    Works regardless of how the dataframe index looks.
    """
    if use_predefined_split and 'split' in self.dataset.columns:
      train_mask = self.dataset['split'].to_numpy() == 'train'
      test_mask = ~train_mask
    else:
      N = len(self.dataset)
      rng = np.random.RandomState(42)
      perm = rng.permutation(N)
      split = int(0.7 * N)
      train_mask = np.zeros(N, dtype=bool)
      train_mask[perm[:split]] = True
      test_mask = ~train_mask

    train_idx = np.flatnonzero(train_mask)  # positional indices
    test_idx = np.flatnonzero(test_mask)
    logger.info('Split: %d train | %d test', len(train_idx), len(test_idx))
    assert not np.intersect1d(train_idx, test_idx).size, 'train/test overlap'
    assert len(train_idx) + len(test_idx) == len(self.dataset), 'missing rows'

    return train_idx, test_idx

  def evaluate_generalization(
    self, decoder_results: Dict[str, Any], threshold: float = 0.5
  ) -> pd.DataFrame:
    """
    Evaluate generalization performance by analyzing successful vs. failed predictions.

    Args:
        decoder_results: Results from train_decoder_with_split
        threshold: Threshold for binary classification

    Returns:
        DataFrame with evaluation results
    """
    best_layer = max(
      decoder_results.keys(), key=lambda k: decoder_results[k]['test_f1_score']
    )
    results = decoder_results[best_layer]

    logger.info(
      f"Evaluating generalization using layer {best_layer} "
      f"(Test F1 = {results['test_f1_score']:.4f}, "
      f"Train F1 = {results['train_f1_score']:.4f})"
    )

    test_indices = results['test_indices']
    predictions = results['probabilities_test']
    ground_truth = results['ground_truth']

    binary_predictions = (predictions > threshold).astype(int)

    correct_components = (binary_predictions == ground_truth).mean(axis=1)

    results_df = pd.DataFrame(
      {
        'index': test_indices,
        'correct_fraction': correct_components,
        'fully_correct': (binary_predictions == ground_truth).all(axis=1),
        'ground_truth': list(ground_truth),
        'prediction': list(predictions),
        'binary_prediction': list(binary_predictions),
      }
    )

    subset = self.dataset.iloc[test_indices].reset_index(drop=True)
    results_df['prompt'] = subset['prompt'].values
    results_df['image_path'] = subset['image_path'].values

    if self.prompt_generator is not None:
      results_df['components'] = [
        self.prompt_generator.get_components_from_latent(gt) for gt in ground_truth
      ]
      results_df['predicted_components'] = [
        self.prompt_generator.get_components_from_latent(pred)
        for pred in binary_predictions
      ]

    results_df = results_df.sort_values('correct_fraction', ascending=False)

    model_dir = self.get_model_dir()
    results_df.to_csv(model_dir / 'generalization_evaluation.csv', index=False)

    logger.info(
      f"Successful predictions: {results_df['fully_correct'].sum()} / {len(results_df)} ="
      f" {results_df['fully_correct'].mean():.2%}"
    )

    logger.info(
      f'Average component accuracy: {results_df["correct_fraction"].mean():.2%}'
    )

    # Save misclassified images to a separate folder
    mistakes_dir = model_dir / 'misclassified'
    mistakes_dir.mkdir(exist_ok=True, parents=True)

    misclassified = results_df[~results_df['fully_correct']]

    if len(misclassified) > 0:
      for i, row in enumerate(misclassified.itertuples(index=False)):
        src_path = row.image_path
        dst_path = mistakes_dir / f'mistake_{i}.png'

        try:
          shutil.copy2(src_path, dst_path)
        except Exception as e:
          logger.error(f'Error copying {src_path} to {dst_path}: {e}')

    return results_df

  def rank_ood_samples(
    self,
    decoder_results: Dict[str, Any],
    metric_name: str = 'mse',
    clear_existing: bool = True,
    *,
    write_images: bool = True,
  ) -> pd.DataFrame:
    """
    Rank OOD samples for the *best* layer and optionally copy the
    ranked / mis-classified images.

    If ``write_images`` is False we still produce a CSV, but we skip
    the (slow / space-hungry) image copies.
    """
    best_layer = max(
      decoder_results.keys(), key=lambda k: decoder_results[k]['test_f1_score']
    )
    results = decoder_results[best_layer]

    logger.info(
      f"Ranking OOD samples using layer {best_layer} "
      f"(Test f1 = {results['test_f1_score']:.4f}) with metric: {metric_name}"
    )

    test_indices = results['test_indices']
    predictions = results['probabilities_test']
    ground_truth = results['ground_truth']

    metric_fn = get_metric(metric_name)
    metric_values = metric_fn(predictions, ground_truth)

    ranking_df = pd.DataFrame(
      {
        'index': test_indices,
        metric_name: metric_values,
        'ground_truth': list(ground_truth),
        'prediction': list(predictions),
      }
    )

    subset = self.dataset.iloc[test_indices].reset_index(drop=True)
    ranking_df['prompt'] = subset['prompt'].values
    ranking_df['image_path'] = subset['image_path'].values

    ascending = metric_name in {'mse', 'mae', 'logloss'}
    ranking_df = ranking_df.sort_values(metric_name, ascending=ascending)

    model_dir = self.get_model_dir()

    if write_images:
      ranked_dir = model_dir / f'ood_ranked_{metric_name}'
      if clear_existing and ranked_dir.exists():
        clear_directory(ranked_dir)
      ranked_dir.mkdir(exist_ok=True, parents=True)

      for i, (_, row) in enumerate(ranking_df.iterrows(), 1):
        self._save_ranked_image(ranked_dir, i, row)

      ranking_df.to_csv(ranked_dir / 'rankings.csv', index=False)
    else:
      csv_dir = model_dir / 'rankings'
      csv_dir.mkdir(exist_ok=True, parents=True)
      ranking_df.to_csv(csv_dir / f'rankings_{metric_name}_best.csv', index=False)

    return ranking_df

  def rank_layers(
    self,
    decoder_results: Dict[str, Any],
    metric_name: str = 'mse',
    *,
    write_images: bool = True,
  ) -> Dict[str, pd.DataFrame]:
    """
    Rank **every** layer with the same metric.

    Returns
    -------
    dict
        mapping  ``layer_name → ranking_dataframe``.
    """
    model_dir = self.get_model_dir()
    csv_parent = model_dir / 'rankings'
    csv_parent.mkdir(exist_ok=True, parents=True)

    ordered = sorted(
      decoder_results.items(),
      key=lambda kv: kv[1]['test_f1_score'],
      reverse=True,
    )

    out: Dict[str, pd.DataFrame] = {}
    for layer, res in ordered:
      logger.info(f'Ranking layer {layer} …')
      df = self.rank_ood_samples(
        {layer: res},
        metric_name=metric_name,
        clear_existing=False,
        write_images=write_images,
      )
      df.to_csv(csv_parent / f'rankings_{layer}.csv', index=False)
      out[layer] = df

    return out

  def _save_ranked_image(self, ranked_dir: Path, rank: int, row: pd.Series) -> None:
    """
    Save a ranked image with a descriptive filename.

    Args:
        ranked_dir: Directory to save ranked images
        rank: Rank of this image (1 = best)
        row: DataFrame row with image data
    """
    import shutil

    src_path = row['image_path']

    animal_names = self._extract_animal_names(row)

    if animal_names:
      animals_str = '_'.join(animal_names)
      dst_path = ranked_dir / f'{rank}_{animals_str}.png'
    else:
      dst_path = ranked_dir / f'{rank}.png'

    try:
      shutil.copy2(src_path, dst_path)
    except Exception as e:
      logger.error(f'Error copying {src_path} to {dst_path}: {e}')

  def _extract_animal_names(self, row: pd.Series) -> List[str]:
    """
    Extract animal names from either components or prompt.

    Args:
        row: DataFrame row with prompt and ground truth

    Returns:
        List of animal names
    """
    animal_names = []

    if self.prompt_generator is not None:
      try:
        components = self.prompt_generator.get_components_from_latent(row['ground_truth'])
        for comp_elements in components.values():
          for element in comp_elements:
            clean_element = element.replace('one ', '').replace(' ', '')
            animal_names.append(clean_element)
      except Exception as e:
        logger.debug(f'Failed to extract components: {e}')

    if not animal_names:
      prompt = row['prompt']
      if 'with ' in prompt:
        animals_part = prompt.split('with ')[1]
        animal_parts = [p.strip() for p in animals_part.split(',')]
        for part in animal_parts:
          clean_animal = part.replace('one ', '').replace(' ', '')
          animal_names.append(clean_animal)

    return animal_names

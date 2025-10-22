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
Configuration module for compositional generalization analysis.

This module provides component configurations for different compositional setups
and utility functions for configuration management.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .data import ComponentConfig


@dataclass
class AnalysisConfig:
  """Configuration for analysis experiments."""

  model_name: str
  compositional_setup: str
  output_dir: Path
  cache_dir: Optional[Path] = None
  hotness: int = 3
  variable_hotness: bool = False
  max_samples: Optional[int] = None
  force_recompute: bool = False
  specific_layer: Optional[str] = None
  analyze_all_layers: bool = False
  metric: str = 'mse'
  num_seeds: int = 1

  @property
  def model_dir(self) -> Path:
    """Get the directory for the current model."""
    model_dir_name = self.model_name.replace('/', '_')
    return self.output_dir / model_dir_name / self.compositional_setup


# Available models
AVAILABLE_MODELS = [
  'runwayml/stable-diffusion-v1-5',
  'stabilityai/stable-diffusion-xl-base-1.0',
  'stabilityai/stable-diffusion-3-medium-diffusers',
  'stabilityai/stable-diffusion-3.5-medium',
  'stabilityai/stable-diffusion-3.5-large',
  'black-forest-labs/FLUX.1-dev',
  'black-forest-labs/FLUX.1-schnell',
]


# Define component configurations for different compositional setups
COMPONENT_CONFIGS = {
  # Theory single-component configs
  'animals_short': ComponentConfig(
    name='animals',
    elements=['giraffe', 'lion', 'elephant', 'crocodile'],
    prompt_template='Looking at this image, identify which animal(s) are present. '
    'Analyze and return a boolean value (true/false) for each of these specific animals: '
    'giraffe, lion, elephant, and crocodile. '
    'Return your answer in JSON format where each animal is a boolean property.',
  ),
  'animals': ComponentConfig(
    name='animals',
    elements=[
      'giraffe',
      'lion',
      'elephant',
      'crocodile',
      'bear',
      'snake',
      'eagle',
      'cow',
    ],
    prompt_template='Look at this image and identify which animals are present. '
    'For each animal in this list (giraffe, lion, elephant, crocodile, bear, snake, eagle, cow), '
    'indicate with a boolean value (true/false) whether it appears in the image. '
    'Return your analysis as a JSON object with each animal as a property.',
  ),
  'animals_long': ComponentConfig(
    name='animals',
    elements=[
      'giraffe',
      'lion',
      'elephant',
      'crocodile',
      'bear',
      'snake',
      'eagle',
      'cow',
      'zebra',
      'tiger',
      'rhino',
      'hippo',
      'wolf',
      'fox',
      'deer',
      'monkey',
      'panda',
      'koala',
      'kangaroo',
      'penguin',
    ],
    prompt_template='Analyze this image and identify which animals are present from the following comprehensive list: '
    'giraffe, lion, elephant, crocodile, bear, snake, eagle, cow, zebra, tiger, rhino, hippo, wolf, '
    'fox, deer, monkey, panda, koala, kangaroo, and penguin. '
    'For each animal, return a boolean value (true if present, false if not) in your JSON response.',
  ),
  'objects': ComponentConfig(
    name='objects',
    elements=[
      'red cube',
      'blue sphere',
      'green cylinder',
      'yellow cone',
      'purple pyramid',
      'orange triangular prism',
      'black cuboid',
      'brown triangular prism',
    ],
    prompt_template='Examine this image and identify which geometric objects are present. '
    'For each of these specific objects (red cube, blue sphere, green cylinder, yellow cone, '
    'purple pyramid, orange triangular prism, black cuboid, brown triangular prism), '
    'indicate with a boolean value (true/false) whether it appears in the image. '
    'Return your analysis as a JSON object with each object as a property.',
  ),
  'objects_two_colours': ComponentConfig(
    name='objects',
    elements=[
      'red cube',
      'green cube',
      'blue sphere',
      'orange sphere',
      'green cylinder',
      'purple cylinder',
      'yellow cone',
      'blue cone',
      'purple pyramid',
      'red pyramid',
      'orange cuboid',
      'yellow cuboid',
    ],
    prompt_template='Carefully examine this image and identify which colored geometric objects are present. '
    'For each specific color-shape combination (red cube, green cube, blue sphere, orange sphere, '
    'green cylinder, purple cylinder, yellow cone, blue cone, purple pyramid, red pyramid, '
    'orange cuboid, yellow cuboid), indicate with a boolean value (true/false) whether it appears in the image. '
    'Pay close attention to both the shape AND color of each object. '
    'Return your analysis as a JSON object with each colored shape as a property.',
  ),
  'objects_three_colours': ComponentConfig(
    name='objects',
    elements=[
      'red cube',
      'green cube',
      'blue cube',
      'blue sphere',
      'orange sphere',
      'red sphere',
      'green cylinder',
      'purple cylinder',
      'yellow cylinder',
      'yellow cone',
      'blue cone',
      'green cone',
      'purple pyramid',
      'red pyramid',
      'orange pyramid',
      'orange cuboid',
      'yellow cuboid',
      'blue cuboid',
    ],
    prompt_template='Analyze this image and identify which colored 3D geometric shapes are present. '
    'The image may contain various shapes (cubes, spheres, cylinders, cones, pyramids, cuboids) '
    'in different colors (red, green, blue, yellow, orange, purple). '
    'For each specific color-shape combination listed in the schema, '
    'indicate with a boolean value (true/false) whether it appears in the image. '
    'Be precise about both the exact shape AND its color. '
    'Return your analysis as a structured JSON object.',
  ),
  'fruits_veggies_long': ComponentConfig(
    name='fruits_veggies',
    elements=[
      'apple',
      'banana',
      'orange',
      'grape',
      'strawberry',
      'watermelon',
      'pineapple',
      'mango',
      'blueberry',
      'peach',
      'lemon',
      'cherry',
      'carrot',
      'broccoli',
      'tomato',
      'cucumber',
      'potato',
      'onion',
      'pepper',
      'lettuce',
    ],
    prompt_template='Examine this image and identify which fruits and vegetables are present. '
    'For each item in this list (apple, banana, orange, grape, strawberry, watermelon, pineapple, mango, '
    'blueberry, peach, lemon, cherry, carrot, broccoli, tomato, cucumber, potato, onion, pepper, lettuce), '
    'indicate with a boolean value (true/false) whether it appears in the image. '
    'Return your analysis as a JSON object with each fruit or vegetable as a property.',
  ),
  'animals_with_clothes': ComponentConfig(
    name='animals_with_clothes',
    elements=[],
    template='{animal} wearing {clothing}',
    prompt_template='This image shows an animal wearing an item of clothing. '
    'Identify both: \n'
    '1. Which animal is shown (choose one from: cat, dog, bear, lion, elephant, giraffe, monkey, zebra, tiger, panda)\n'
    '2. What clothing item the animal is wearing (choose one from: hat, sunglasses, scarf, bowtie, jacket, crown, tie, cape, sweater, necklace)\n'
    "Return your answer as a JSON object with 'animal' and 'clothing' properties, using exactly one of the provided options for each.",
    components=[
      ComponentConfig(
        name='animal',
        elements=[
          'A cat',
          'A dog',
          'A bear',
          'A lion',
          'An elephant',
          'A giraffe',
          'A monkey',
          'A zebra',
          'A tiger',
          'A panda',
        ],
      ),
      ComponentConfig(
        name='clothing',
        elements=[
          'a hat',
          'a pair of sunglasses',
          'a scarf',
          'a bowtie',
          'a jacket',
          'a crown',
          'a tie',
          'a cape',
          'a sweater',
          'a necklace',
        ],
      ),
    ],
  ),
  'animals_with_clothes_and_food': ComponentConfig(
    name='animals_with_clothes_and_food',
    elements=[],
    template='{animal} wearing {clothing} eating {food}',
    prompt_template='This image shows an animal wearing clothing and eating food. '
    'Identify all three elements:\n'
    '1. Which animal is shown (choose one from: cat, dog, bear, lion, elephant, giraffe, monkey, zebra, tiger, panda)\n'
    '2. What clothing item the animal is wearing (choose one from: hat, sunglasses, scarf, bowtie, jacket, crown, tie, cape, pants, boots)\n'
    '3. What food the animal is eating (choose one from: pizza, banana, ice cream, cake, hamburger, apple, watermelon, donut, sandwich, salad)\n'
    "Return your answer as a JSON object with 'animal', 'clothing', and 'food' properties, using exactly one of the provided options for each category.",
    components=[
      ComponentConfig(
        name='animal',
        elements=[
          'A cat',
          'A dog',
          'A bear',
          'A lion',
          'An elephant',
          'A giraffe',
          'A monkey',
          'A zebra',
          'A tiger',
          'A panda',
        ],
      ),
      ComponentConfig(
        name='clothing',
        elements=[
          'a hat',
          'a pair of sunglasses',
          'a scarf',
          'a bowtie',
          'a jacket',
          'a crown',
          'a tie',
          'a cape',
          'a pair of pants',
          'a pair of boots',
        ],
      ),
      ComponentConfig(
        name='food',
        elements=[
          'a pizza',
          'a banana',
          'an ice cream',
          'a cake',
          'a hamburger',
          'an apple',
          'a watermelon',
          'a donut',
          'a sandwich',
          'a salad',
        ],
      ),
    ],
  ),
  'animals_with_food_eyes_and_clothes': ComponentConfig(
    name='animals_with_food_eyes_and_clothes',
    elements=[],
    template='A {animal} with {food} as eyes wearing {clothing}',
    prompt_template='This image shows a surreal scene with an animal that has food items as eyes and is wearing clothing. '
    'Identify the following three elements:\n'
    '1. Which animal is shown (choose one from: cat, bear, lion, elephant, giraffe, monkey, zebra, panda, wolf, rabbit)\n'
    "2. What food items are used as the animal's eyes (choose one from: strawberries, oranges, burgers, watermelons, donuts, cookies, cupcakes, pizza, lemons, tomatoes)\n"
    '3. What clothing item the animal is wearing (choose one from: crown, cowboy hat, scarf, cape, hawaiian shirt, leather jacket, pants, tuxedo, raincoat, sunglasses)\n'
    "Return your answer as a JSON object with 'animal', 'food', and 'clothing' properties, using exactly one of the provided options for each category.",
    components=[
      ComponentConfig(
        name='animal',
        elements=[
          'A cat',
          'A bear',
          'A lion',
          'An elephant',
          'A giraffe',
          'A monkey',
          'A zebra',
          'A panda',
          'A wolf',
          'A rabbit',
        ],
      ),
      ComponentConfig(
        name='food',
        elements=[
          'strawberries',
          'oranges',
          'burgers',
          'watermelons',
          'donuts',
          'cookies',
          'cupcakes',
          'pizza',
          'lemons',
          'tomatoes',
        ],
      ),
      ComponentConfig(
        name='clothing',
        elements=[
          'a crown',
          'a cowboy hat',
          'a scarf',
          'a cape',
          'a hawaiian shirt',
          'a leather jacket',
          'a pair of pants',
          'a tuxedo',
          'a raincoat',
          'sunglasses',
        ],
      ),
    ],
  ),
  'counting_objects': ComponentConfig(
    name='counting_objects',
    elements=[],
    template='An image with exactly {number} {object}',
    prompt_template='This image shows multiple instances of a single type of object. '
    'Your task is to:\n'
    '1. Identify what type of object is shown (select from: tomatoes, onions, oranges, wolves, bears, apples, bananas, '
    'carrots, cucumbers, strawberries, lemons, cherries, grapes, peaches, pears, foxes, rabbits, cats, dogs, sheep)\n'
    '2. Count EXACTLY how many of these objects are present in the image (select from: one, two, three, four, five, six, seven, '
    'eight, nine, ten, eleven, twelve, thirteen, fourteen, fifteen, sixteen, seventeen, eighteen, nineteen, twenty)\n'
    "Return your answer as a JSON object with the 'number' and 'object' properties. Be precise in your counting - "
    'count each instance carefully and select the exact number word that matches your count. '
    "For example, if you see 7 apples, you should return 'seven' for number and 'apples' for object.",
    components=[
      ComponentConfig(
        name='number',
        elements=[
          'one',
          'two',
          'three',
          'four',
          'five',
          'six',
          'seven',
          'eight',
          'nine',
          'ten',
          'eleven',
          'twelve',
          'thirteen',
          'fourteen',
          'fifteen',
          'sixteen',
          'seventeen',
          'eighteen',
          'nineteen',
          'twenty',
        ],
      ),
      ComponentConfig(
        name='object',
        elements=[
          'tomatoes',
          'onions',
          'oranges',
          'wolves',
          'bears',
          'apples',
          'bananas',
          'carrots',
          'cucumbers',
          'strawberries',
          'lemons',
          'cherries',
          'grapes',
          'peaches',
          'pears',
          'foxes',
          'rabbits',
          'cats',
          'dogs',
          'sheep',
        ],
      ),
    ],
  ),
  'impossible_materials': ComponentConfig(
    name='impossible_materials',
    elements=[],
    template='A {object} made entirely of {material} sitting on a {surface}',
    prompt_template='This image shows a surreal scene with an object made of an unusual material, sitting on an unusual surface. '
    'Identify all three elements:\n'
    '1. What object is depicted (choose one from: chair, bicycle, bookshelf, piano, computer, refrigerator, watch, umbrella, camera, guitar)\n'
    '2. What unusual material the object appears to be made of (choose one from: liquid water, fire, smoke, mirrors, ice, tree bark, glass noodles, gelatin, paper, soap bubbles)\n'
    '3. What unusual surface the object is sitting on (choose one from: clouds, ocean waves, melting ice, sand dunes, moss, broken glass, spiderwebs, lily pads, autumn leaves, foam)\n'
    "Return your answer as a JSON object with 'object', 'material', and 'surface' properties, using exactly one of the provided options for each category.",
    components=[
      ComponentConfig(
        name='object',
        elements=[
          'chair',
          'bicycle',
          'bookshelf',
          'piano',
          'computer',
          'refrigerator',
          'watch',
          'umbrella',
          'camera',
          'guitar',
        ],
      ),
      ComponentConfig(
        name='material',
        elements=[
          'liquid water',
          'fire',
          'smoke',
          'mirrors',
          'ice',
          'tree bark',
          'glass noodles',
          'gelatin',
          'paper',
          'soap bubbles',
        ],
      ),
      ComponentConfig(
        name='surface',
        elements=[
          'clouds',
          'ocean waves',
          'melting ice',
          'sand dunes',
          'moss',
          'broken glass',
          'spiderwebs',
          'lily pads',
          'autumn leaves',
          'foam',
        ],
      ),
    ],
  ),
  'nested_containment': ComponentConfig(
    name='nested_containment',
    elements=[],
    template='A {container1} containing a {container2} containing a {object}',
    prompt_template='This image shows a nested containment scene with three levels. '
    'Identify each level of containment from outside to inside:\n'
    '1. Outer container (choose one from: transparent cube, glass jar, wooden box, metal safe, woven basket, leather bag, ceramic pot, copper kettle, crystal sphere, rubber ball)\n'
    '2. Middle container (choose one from: small treasure chest, porcelain teacup, silk pouch, stone bowl, paper envelope, cardboard tube, tin can, shell, gold locket, velvet case)\n'
    '3. Inner object (choose one from: diamond, living butterfly, ticking clock, miniature planet, flickering flame, drop of mercury, hologram, glowing ember, snowflake, single cell organism)\n'
    "Return your answer as a JSON object with 'container1', 'container2', and 'object' properties, using exactly one of the provided options for each level of containment.",
    components=[
      ComponentConfig(
        name='container1',
        elements=[
          'transparent cube',
          'glass jar',
          'wooden box',
          'metal safe',
          'woven basket',
          'leather bag',
          'ceramic pot',
          'copper kettle',
          'crystal sphere',
          'rubber ball',
        ],
      ),
      ComponentConfig(
        name='container2',
        elements=[
          'small treasure chest',
          'porcelain teacup',
          'silk pouch',
          'stone bowl',
          'paper envelope',
          'cardboard tube',
          'tin can',
          'shell',
          'gold locket',
          'velvet case',
        ],
      ),
      ComponentConfig(
        name='object',
        elements=[
          'diamond',
          'living butterfly',
          'ticking clock',
          'miniature planet',
          'flickering flame',
          'drop of mercury',
          'hologram',
          'glowing ember',
          'snowflake',
          'single cell organism',
        ],
      ),
    ],
  ),
  'animals_vegetables_shapes': ComponentConfig(
    name='animals_vegetables_shapes',
    elements=[
      'lion',
      'elephant',
      'giraffe',
      'tiger',
      'bear',
      'zebra',
      'monkey',
      'carrot',
      'broccoli',
      'potato',
      'tomato',
      'cucumber',
      'onion',
      'pepper',
      'cube',
      'sphere',
      'cylinder',
      'cone',
      'pyramid',
      'triangular prism',
    ],
    prompt_template='Examine this image and identify which elements are present. '
    'For each item in this list (lion, elephant, giraffe, tiger, bear, zebra, monkey, '
    'carrot, broccoli, potato, tomato, cucumber, onion, pepper, '
    'cube, sphere, cylinder, cone, pyramid, triangular prism), '
    'indicate with a boolean value (true/false) whether it appears in the image. '
    'Return your analysis as a JSON object with each element as a property.',
  ),
  'stacked_foods': ComponentConfig(
    name='stacked_foods',
    elements=[],
    template='{food_top} on top of {food_middle} on top of {food_bottom}',
    prompt_template=(
      'This image shows three foods stacked vertically.  Identify **each layer**:\n'
      '1. The food on the very top (choose one from: burger, pizza, salad, sushi, taco, donut, ice cream, pancake, spaghetti, sandwich)\n'
      '2. The food in the middle (same options)\n'
      '3. The food on the bottom (same options)\n\n'
      "Return a JSON object with the keys **'food_top'**, **'food_middle'**, and **'food_bottom'**, "
      'each containing exactly one of the allowed food names.'
    ),
    components=[
      ComponentConfig(
        name='food_top',
        elements=[
          'A burger',
          'A pizza',
          'A salad',
          'Sushi',
          'A taco',
          'A donut',
          'Ice cream',
          'A pancake',
          'Spaghetti',
          'A sandwich',
        ],
      ),
      ComponentConfig(
        name='food_middle',
        elements=[
          'a burger',
          'a pizza',
          'a salad',
          'sushi',
          'a taco',
          'a donut',
          'ice cream',
          'a pancake',
          'spaghetti',
          'a sandwich',
        ],
      ),
      ComponentConfig(
        name='food_bottom',
        elements=[
          'a burger',
          'a pizza',
          'a salad',
          'sushi',
          'a taco',
          'a donut',
          'ice cream',
          'a pancake',
          'spaghetti',
          'a sandwich',
        ],
      ),
    ],
  ),
  'animals_chasing_chain': ComponentConfig(
    name='animals_chasing_chain',
    elements=[],
    template='{animal1} chasing {animal2} chasing {animal3}',
    prompt_template=(
      'This image shows a *chase chain* of two **or** three animals.  '
      'All animals are chosen from the following list:\n'
      'lion, elephant, giraffe, crocodile, bear, snake, eagle, cow, zebra, tiger.\n\n'
      'Identify which animal appears in each ordered slot:\n'
      '• **animal1** – the one doing the chasing at the back of the chain\n'
      '• **animal2** – the one being chased (and possibly chasing another)\n'
      '• **animal3** – the animal at the very front that is only being chased\n\n'
      'Return a JSON object with exactly these keys:\n'
      '```json\n'
      '{\n'
      '  "animal1": "<animal name>",\n'
      '  "animal2": "<animal name>",\n'
      '  "animal3": "<animal name>",  \n'
      '}\n'
    ),
    components=[
      ComponentConfig(
        name='animal1',
        elements=[
          'A lion',
          'An elephant',
          'A giraffe',
          'A crocodile',
          'A bear',
          'A snake',
          'An eagle',
          'A cow',
          'A zebra',
          'A tiger',
        ],
      ),
      ComponentConfig(
        name='animal2',
        elements=[
          'a lion',
          'an elephant',
          'a giraffe',
          'a crocodile',
          'a bear',
          'a snake',
          'an eagle',
          'a cow',
          'a zebra',
          'a tiger',
        ],
      ),
      ComponentConfig(
        name='animal3',
        elements=[
          'a lion',
          'an elephant',
          'a giraffe',
          'a crocodile',
          'a bear',
          'a snake',
          'an eagle',
          'a cow',
          'a zebra',
          'a tiger',
        ],
      ),
    ],
  ),
  'stacked_animals': ComponentConfig(
    name='stacked_animals',
    elements=[],
    template='{animal_top} on top of {animal_middle} on top of {animal_bottom}',
    prompt_template=(
      'This image shows three animals stacked vertically.  Identify **each layer**:\n'
      '1. The animal on the very top (choose one from: lion, elephant, giraffe, crocodile, '
      'bear, snake, eagle, cow, zebra, tiger)\n'
      '2. The animal in the middle (same options)\n'
      '3. The animal on the bottom (same options)\n\n'
      "Return a JSON object with the keys **'animal_top'**, **'animal_middle'**, and "
      "**'animal_bottom'**, each containing exactly one of the allowed animal names."
    ),
    components=[
      ComponentConfig(
        name='animal_top',
        elements=[
          'A lion',
          'An elephant',
          'A giraffe',
          'A crocodile',
          'A bear',
          'A snake',
          'An eagle',
          'A cow',
          'A zebra',
          'A tiger',
        ],
      ),
      ComponentConfig(
        name='animal_middle',
        elements=[
          'a lion',
          'an elephant',
          'a giraffe',
          'a crocodile',
          'a bear',
          'a snake',
          'an eagle',
          'a cow',
          'a zebra',
          'a tiger',
        ],
      ),
      ComponentConfig(
        name='animal_bottom',
        elements=[
          'a lion',
          'an elephant',
          'a giraffe',
          'a crocodile',
          'a bear',
          'a snake',
          'an eagle',
          'a cow',
          'a zebra',
          'a tiger',
        ],
      ),
    ],
  ),
  'animals_with_adjectives': ComponentConfig(
    name='animals_with_adjectives',
    elements=[],
    template='A image of  a {adjective} {animal}',
    prompt_template=(
      'This image shows an animal with a certain characteristic, performing an action. Identify:\n'
      "1. The animal's characteristic/emotion (adjective) (choose from: happy, sad, angry, sleepy, curious, playful, scared, proud, surprised, bored)\n"
      '2. The animal (choose from: lion, elephant, giraffe, crocodile, bear, snake, eagle, cow, zebra, tiger)\n'
      "Return your answer as a JSON object with 'adjective' and 'animal' properties."
    ),
    components=[
      ComponentConfig(
        name='adjective',
        elements=[
          'happy',
          'sad',
          'angry',
          'sleepy',
          'curious',
          'playful',
          'scared',
          'proud',
          'surprised',
          'bored',
        ],
      ),
      ComponentConfig(
        name='animal',
        elements=[
          'lion',
          'elephant',
          'giraffe',
          'crocodile',
          'bear',
          'snake',
          'eagle',
          'cow',
          'zebra',
          'tiger',
        ],
      ),
    ],
  ),
  'nested_containment_animals': ComponentConfig(
    name='nested_containment_animals',
    elements=[],
    template='A {animal_outer} containing a {animal_middle} containing an {animal_inner}',
    prompt_template=(
      'This image depicts a surreal scene of animals nested within each other, like Matryoshka dolls. Identify:\n'
      '1. The outermost animal (container 1) (choose from: lion, elephant, giraffe, crocodile, bear, snake, eagle, cow, zebra, tiger)\n'
      '2. The middle animal (container 2) (choose from: lion, elephant, giraffe, crocodile, bear, snake, eagle, cow, zebra, tiger)\n'
      '3. The innermost animal (object) (choose from: lion, elephant, giraffe, crocodile, bear, snake, eagle, cow, zebra, tiger)\n'
      "Return your answer as a JSON object with 'animal_outer', 'animal_middle', and 'animal_inner' properties. Ensure the animals are identified from outermost to innermost."
    ),
    components=[
      ComponentConfig(
        name='animal_outer',
        elements=[
          'A lion',
          'An elephant',
          'A giraffe',
          'A crocodile',
          'A bear',
          'A snake',
          'An eagle',
          'A cow',
          'A zebra',
          'A tiger',
        ],
      ),
      ComponentConfig(
        name='animal_middle',
        elements=[
          'a lion',
          'an elephant',
          'a giraffe',
          'a crocodile',
          'a bear',
          'a snake',
          'an eagle',
          'a cow',
          'a zebra',
          'a tiger',
        ],
      ),
      ComponentConfig(
        name='animal_inner',
        elements=[
          'a lion',
          'an elephant',
          'a giraffe',
          'a crocodile',
          'a bear',
          'a snake',
          'a eagle',
          'a cow',
          'a zebra',
          'a tiger',
        ],
      ),
    ],
  ),
  'three_animals_three_materials': ComponentConfig(
    name='three_animals_three_materials',
    elements=[],
    template='{animal_fire} made of fire, {animal_ice} made of ice, and {animal_wood} made of wood',
    prompt_template=(
      'This image shows three animals, each made from a different unusual material:\n\n'
      '1. One animal is made of **fire**\n'
      '2. One animal is made of **ice**\n'
      '3. One animal is made of **wood**\n\n'
      'All animals are chosen from the following list:\n'
      'lion, elephant, giraffe, crocodile, bear, snake, eagle, cow, zebra, tiger\n\n'
      'Return your answer as a JSON object with exactly these keys:\n'
      '{\n'
      '  "animal_fire": "<animal name>",\n'
      '  "animal_ice": "<animal name>",\n'
      '  "animal_wood": "<animal name>"\n'
      '}\n\n'
      'Each animal must be chosen from the list and appear only once across the three roles.'
    ),
    components=[
      ComponentConfig(
        name='animal_fire',
        elements=[
          'A lion',
          'An elephant',
          'A giraffe',
          'A crocodile',
          'A bear',
          'A snake',
          'An eagle',
          'A cow',
          'A zebra',
          'A tiger',
        ],
      ),
      ComponentConfig(
        name='animal_ice',
        elements=[
          'a lion',
          'an elephant',
          'a giraffe',
          'a crocodile',
          'a bear',
          'a snake',
          'an eagle',
          'a cow',
          'a zebra',
          'a tiger',
        ],
      ),
      ComponentConfig(
        name='animal_wood',
        elements=[
          'a lion',
          'an elephant',
          'a giraffe',
          'a crocodile',
          'a bear',
          'a snake',
          'a eagle',
          'a cow',
          'a zebra',
          'a tiger',
        ],
      ),
    ],
  ),
  'three_animals_three_verbs': ComponentConfig(
    name='three_animals_three_verbs',
    elements=[],
    template='{animal_singing} singing, {animal_eating} eating, and {animal_sleeping} sleeping',
    prompt_template=(
      'This image shows three animals, each performing a different action:\n\n'
      '1. One animal is **singing**\n'
      '2. One animal is **eating**\n'
      '3. One animal is **sleeping**\n\n'
      'All animals are chosen from the following list:\n'
      'lion, elephant, giraffe, crocodile, bear, snake, eagle, cow, zebra, tiger\n\n'
      'Return your answer as a JSON object with exactly these keys:\n'
      '{\n'
      '  "animal_singing": "<animal name>",\n'
      '  "animal_eating": "<animal name>",\n'
      '  "animal_sleeping": "<animal name>"\n'
      '}\n\n'
      'Each animal must appear only once and be matched to exactly one verb.'
    ),
    components=[
      ComponentConfig(
        name='animal_singing',
        elements=[
          'A lion',
          'An elephant',
          'A giraffe',
          'A crocodile',
          'A bear',
          'A snake',
          'An eagle',
          'A cow',
          'A zebra',
          'A tiger',
        ],
      ),
      ComponentConfig(
        name='animal_eating',
        elements=[
          'a lion',
          'an elephant',
          'a giraffe',
          'a crocodile',
          'a bear',
          'a snake',
          'an eagle',
          'a cow',
          'a zebra',
          'a tiger',
        ],
      ),
      ComponentConfig(
        name='animal_sleeping',
        elements=[
          'a lion',
          'an elephant',
          'a giraffe',
          'a crocodile',
          'a bear',
          'a snake',
          'a eagle',
          'a cow',
          'a zebra',
          'a tiger',
        ],
      ),
    ],
  ),
  'three_animals_three_adjectives': ComponentConfig(
    name='three_animals_three_adjectives',
    elements=[],
    template='A sad {animal_happy}, a happy {animal_sad}, and an angry {animal_angry}',
    prompt_template=(
      'This image depicts three animals, each showing a different emotion:\n\n'
      '1. One is **happy**\n'
      '2. One is **sad**\n'
      '3. One is **angry**\n\n'
      'All animals are chosen from this list:\n'
      'lion, elephant, giraffe, crocodile, bear, snake, eagle, cow, zebra, tiger\n\n'
      'Return your answer as a JSON object with exactly these keys:\n'
      '{\n'
      '  "animal_happy": "<animal name>",\n'
      '  "animal_sad": "<animal name>",\n'
      '  "animal_angry": "<animal name>"\n'
      '}\n\n'
      'Each animal should appear only once across the three roles.'
    ),
    components=[
      ComponentConfig(
        name='animal_happy',
        elements=[
          'lion',
          'elephant',
          'giraffe',
          'crocodile',
          'bear',
          'snake',
          'eagle',
          'cow',
          'zebra',
          'tiger',
        ],
      ),
      ComponentConfig(
        name='animal_sad',
        elements=[
          'lion',
          'elephant',
          'giraffe',
          'crocodile',
          'bear',
          'snake',
          'eagle',
          'cow',
          'zebra',
          'tiger',
        ],
      ),
      ComponentConfig(
        name='animal_angry',
        elements=[
          'lion',
          'elephant',
          'giraffe',
          'crocodile',
          'bear',
          'snake',
          'eagle',
          'cow',
          'zebra',
          'tiger',
        ],
      ),
    ],
  ),
  'three_animals_three_fixed_styles': ComponentConfig(
    name='three_animals_three_fixed_styles',
    elements=[],
    template='A {animal_pixel} in pixel art, a {animal_oil} in oil painting, and a {animal_cartoon} in cartoon style',
    prompt_template=(
      'This image shows three animals, each depicted in a different artistic style. '
      'Identify which animal is shown in each style:\n\n'
      '1. One animal is illustrated in **pixel art**\n'
      '2. One animal is illustrated in **oil painting**\n'
      '3. One animal is illustrated in **cartoon style**\n\n'
      'All animals are chosen from this list:\n'
      'lion, elephant, giraffe, crocodile, bear, snake, eagle, cow, zebra, tiger\n\n'
      'Return your answer as a JSON object with exactly these keys:\n'
      '{\n'
      '  "animal_pixel": "<animal name>",\n'
      '  "animal_oil": "<animal name>",\n'
      '  "animal_cartoon": "<animal name>"\n'
      '}\n\n'
      'Each animal must be unique and assigned to exactly one style.'
    ),
    components=[
      ComponentConfig(
        name='animal_pixel',
        elements=[
          'lion',
          'elephant',
          'giraffe',
          'crocodile',
          'bear',
          'snake',
          'eagle',
          'cow',
          'zebra',
          'tiger',
        ],
      ),
      ComponentConfig(
        name='animal_oil',
        elements=[
          'lion',
          'elephant',
          'giraffe',
          'crocodile',
          'bear',
          'snake',
          'eagle',
          'cow',
          'zebra',
          'tiger',
        ],
      ),
      ComponentConfig(
        name='animal_cartoon',
        elements=[
          'lion',
          'elephant',
          'giraffe',
          'crocodile',
          'bear',
          'snake',
          'eagle',
          'cow',
          'zebra',
          'tiger',
        ],
      ),
    ],
  ),
  'three_animals_three_fixed_descriptors': ComponentConfig(
    name='three_animals_three_fixed_descriptors',
    elements=[],
    template='A {animal_furry} with fur, a {animal_scaly} with scales, and a {animal_feathered} with feathers',
    prompt_template=(
      'This image shows three animals, each characterized by a distinct type of outer surface:\n\n'
      '1. One animal has **fur** (furry)\n'
      '2. One animal has **scales** (scaly)\n'
      '3. One animal has **feathers** (feathered)\n\n'
      'All animals are chosen from this list:\n'
      'lion, elephant, giraffe, crocodile, bear, snake, eagle, cow, zebra, tiger\n\n'
      'Return your answer as a JSON object with exactly these keys:\n'
      '{\n'
      '  "animal_furry": "<animal name>",\n'
      '  "animal_scaly": "<animal name>",\n'
      '  "animal_feathered": "<animal name>"\n'
      '}\n\n'
      'Each animal must be unique and assigned to exactly one descriptor.'
    ),
    components=[
      ComponentConfig(
        name='animal_furry',
        elements=[
          'lion',
          'elephant',
          'giraffe',
          'crocodile',
          'bear',
          'snake',
          'eagle',
          'cow',
          'zebra',
          'tiger',
        ],
      ),
      ComponentConfig(
        name='animal_scaly',
        elements=[
          'lion',
          'elephant',
          'giraffe',
          'crocodile',
          'bear',
          'snake',
          'eagle',
          'cow',
          'zebra',
          'tiger',
        ],
      ),
      ComponentConfig(
        name='animal_feathered',
        elements=[
          'lion',
          'elephant',
          'giraffe',
          'crocodile',
          'bear',
          'snake',
          'eagle',
          'cow',
          'zebra',
          'tiger',
        ],
      ),
    ],
  ),
  'animals_with_colours': ComponentConfig(
    name='animals_with_colours',
    elements=[],
    template='A {colour} {animal}',
    prompt_template=(
      'This image shows an animal that has been colored in a specific way. '
      'Identify:\n\n'
      '1. Which animal is shown (choose from: lion, elephant, giraffe, crocodile, bear, snake, eagle, cow, zebra, tiger)\n'
      '2. What color the animal is (choose from: red, blue, green, yellow, orange, purple, pink, brown, black, white)\n\n'
      'Return your answer as a JSON object with the keys:\n'
      '{ "animal": "<animal name>", "colour": "<colour name>" }'
    ),
    components=[
      ComponentConfig(
        name='animal',
        elements=[
          'lion',
          'elephant',
          'giraffe',
          'crocodile',
          'bear',
          'snake',
          'eagle',
          'cow',
          'zebra',
          'tiger',
        ],
      ),
      ComponentConfig(
        name='colour',
        elements=[
          'red',
          'blue',
          'green',
          'yellow',
          'orange',
          'purple',
          'pink',
          'brown',
          'black',
          'white',
        ],
      ),
    ],
  ),
  'animals_with_style': ComponentConfig(
    name='animals_with_style',
    elements=[],
    template='A {animal} illustrated in {style} style',
    prompt_template=(
      'This image shows an animal depicted in a particular artistic style. Identify:\n\n'
      '1. The animal shown (choose from: lion, elephant, giraffe, crocodile, bear, snake, eagle, cow, zebra, tiger)\n'
      '2. The artistic style used (choose from: watercolor, pixel art, oil painting, sketch, cartoon, origami, stained glass, pop art, charcoal, clay sculpture)\n\n'
      "Return your answer as a JSON object with keys: 'animal' and 'style'."
    ),
    components=[
      ComponentConfig(
        name='animal',
        elements=[
          'lion',
          'elephant',
          'giraffe',
          'crocodile',
          'bear',
          'snake',
          'eagle',
          'cow',
          'zebra',
          'tiger',
        ],
      ),
      ComponentConfig(
        name='style',
        elements=[
          'watercolor',
          'pixel art',
          'oil painting',
          'sketch',
          'cartoon',
          'origami',
          'stained glass',
          'pop art',
          'charcoal',
          'clay sculpture',
        ],
      ),
    ],
  ),
  'fruits_with_verbs': ComponentConfig(
    name='fruits_with_verbs',
    elements=[],
    template='A {fruit} is {verb}',
    prompt_template=(
      'This image shows a fruit performing an action. Identify:\n\n'
      '1. Which fruit is shown (choose from: apple, banana, orange, grape, strawberry, watermelon, pineapple, mango, blueberry, peach)\n'
      '2. What action (verb) it is performing (choose from: dancing, flying, bouncing, sleeping, swimming, rolling, climbing, stretching, spinning, hiding)\n\n'
      "Return your answer as a JSON object with keys: 'fruit' and 'verb'."
    ),
    components=[
      ComponentConfig(
        name='fruit',
        elements=[
          'apple',
          'banana',
          'orange',
          'grape',
          'strawberry',
          'watermelon',
          'pineapple',
          'mango',
          'blueberry',
          'peach',
        ],
      ),
      ComponentConfig(
        name='verb',
        elements=[
          'dancing',
          'flying',
          'bouncing',
          'sleeping',
          'swimming',
          'rolling',
          'climbing',
          'stretching',
          'spinning',
          'hiding',
        ],
      ),
    ],
  ),
  'animals_with_locations': ComponentConfig(
    name='animals_with_locations',
    elements=[],
    template='A {animal} in the {location}',
    prompt_template=(
      'This image shows a single animal located in a specific part of the scene. Identify:\n\n'
      '1. Which animal is shown (choose from: lion, elephant, giraffe, crocodile, bear, snake, eagle, cow, zebra, tiger)\n'
      '2. Where it appears (choose from: top left, top center, top right, middle left, center, middle right, bottom left, bottom center, bottom right)\n\n'
      "Return your answer as a JSON object with keys: 'animal' and 'location'."
    ),
    components=[
      ComponentConfig(
        name='animal',
        elements=[
          'lion',
          'elephant',
          'giraffe',
          'crocodile',
          'bear',
          'snake',
          'eagle',
          'cow',
          'zebra',
          'tiger',
        ],
      ),
      ComponentConfig(
        name='location',
        elements=[
          'top left',
          'top center',
          'top right',
          'middle left',
          'center',
          'middle right',
          'bottom left',
          'bottom center',
          'bottom right',
        ],
      ),
    ],
  ),
  'counting_animals': ComponentConfig(
    name='counting_animals',
    elements=[],
    template='An image with exactly {number} {animal}',
    prompt_template=(
      'This image shows multiple instances of a single animal species.\n\n'
      'Your task is to:\n'
      '1. **Identify which animal is shown** (choose one from: lion, elephant, giraffe, crocodile, '
      'bear, snake, eagle, cow, zebra, tiger)\n'
      '2. **Count EXACTLY how many of that animal appear** (choose one from: one, two, three, four, '
      'five, six, seven, eight, nine, ten)\n\n'
      "Return your answer as a JSON object with the keys **'number'** and **'animal'**.  "
      'Use the number **word** (e.g. `"seven"`) and the **plural** animal name (e.g. `"lions"`).  '
      'Be precise in your counting—count every individual animal you see.'
    ),
    components=[
      ComponentConfig(
        name='number',
        elements=[
          'one',
          'two',
          'three',
          'four',
          'five',
          'six',
          'seven',
          'eight',
          'nine',
          'ten',
        ],
      ),
      ComponentConfig(
        name='animal',
        elements=[
          'lions',
          'elephants',
          'giraffes',
          'crocodiles',
          'bears',
          'snakes',
          'eagles',
          'cows',
          'zebras',
          'tigers',
        ],
      ),
    ],
  ),
  'animals_with_verbs': ComponentConfig(
    name='animals_with_verbs',
    elements=[],
    template='A {animal} is {verb}',
    prompt_template=(
      'This image shows an animal performing an action. Identify:\n\n'
      '1. Which animal is shown (choose from: lion, elephant, giraffe, crocodile, bear, snake, eagle, cow, zebra, tiger)\n'
      '2. What action (verb) it is performing (choose from: eating, sleeping, running, jumping, flying, swimming, climbing, dancing, playing, hiding)\n\n'
      "Return your answer as a JSON object with keys: 'animal' and 'verb'."
    ),
    components=[
      ComponentConfig(
        name='animal',
        elements=[
          'lion',
          'elephant',
          'giraffe',
          'crocodile',
          'bear',
          'snake',
          'eagle',
          'cow',
          'zebra',
          'tiger',
        ],
      ),
      ComponentConfig(
        name='verb',
        elements=[
          'eating',
          'sleeping',
          'running',
          'jumping',
          'flying',
          'swimming',
          'climbing',
          'dancing',
          'playing',
          'hiding',
        ],
      ),
    ],
  ),
  'animals_with_descriptor': ComponentConfig(
    name='animals_with_descriptor',
    elements=[],
    template='A {descriptor} {animal}',
    prompt_template=(
      'This image shows an animal described by a physical or age trait.\n\n'
      'Animal (choose one): lion, elephant, giraffe, crocodile, bear, snake, '
      'eagle, cow, zebra, tiger\n'
      'Descriptor (choose one): furry, scaly, feathered, leathery, smooth, '
      'wrinkled, young, old, middle-aged, spotted\n\n'
      "Return your answer as a JSON object with keys: 'animal' and 'descriptor'."
    ),
    components=[
      ComponentConfig(
        name='animal',
        elements=[
          'lion',
          'elephant',
          'giraffe',
          'crocodile',
          'bear',
          'snake',
          'eagle',
          'cow',
          'zebra',
          'tiger',
        ],
      ),
      ComponentConfig(
        name='descriptor',
        elements=[
          'furry',
          'scaly',
          'feathered',
          'leathery',
          'smooth',
          'wrinkled',
          'young',
          'old',
          'three-legged',
          'spotted',
        ],
      ),
    ],
  ),
}

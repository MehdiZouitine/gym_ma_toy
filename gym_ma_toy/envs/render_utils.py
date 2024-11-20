import numpy as np
from PIL import Image

from .game_base import MapElement, AuxElement, ElementsColors, AuxElementColors


def render_partially_observable(grid_size, obs, fig_size):
    image = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    image_po = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)

    assert len(MapElement) == len(ElementsColors)
    for element, color in zip(MapElement, ElementsColors):
        image[obs["map"] == element] = color.value
        image_po[obs["partial_map"] == element] = color.value

    image_po[obs["partial_map"] == AuxElement.fog] = AuxElementColors.fog.value

    image = Image.fromarray(image)
    image = image.resize((grid_size * fig_size, grid_size * fig_size), Image.NEAREST)
    image_po = Image.fromarray(image_po)
    image_po = image_po.resize(
        (grid_size * fig_size, grid_size * fig_size), Image.NEAREST
    )
    image = np.array(image, dtype=np.uint8)
    image_po = np.array(image_po, dtype=np.uint8)
    res_image = np.vstack((image_po, image))
    return res_image


def render_observable(grid_size, obs, fig_size):
    image = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)

    assert len(MapElement) == len(ElementsColors)
    for element, color in zip(MapElement, ElementsColors):
        image[obs["map"] == element] = color.value

    image = Image.fromarray(image)
    image = image.resize((grid_size * fig_size, grid_size * fig_size), Image.NEAREST)
    image = np.array(image, dtype=np.uint8)
    return image

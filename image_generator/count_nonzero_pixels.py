import numpy as np
from PIL import Image

# A szkript megszámolja, hogy hány pixel rendelkezik nem nulla intenzitással.

img_name = '256c101'

img = Image.open(f'../test_images/{img_name}.tif')

intensity_array = np.array(img, dtype=np.float64)

intensity_pixel_num = np.count_nonzero(intensity_array > 0.0)

print(f'\nNem nulla intenzitású pixelek száma: {intensity_pixel_num}.')

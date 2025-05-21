import time
import numpy as np
import matplotlib.pyplot as plt
import radon_rect_gpu_processing
from tifffile import imsave
from PIL import Image


# Radon-transzformáció elvégzése rect() bázisfüggvény segítségével, az eredmény vizualizálásával.
if __name__ == '__main__':

    start_time = time.time()
    
    
    image_name = "256c10"
    
    input_img = Image.open(f'../../test_images/{image_name}.tif')
    
    image_array = np.array(input_img, dtype=np.float64)
    side_length = image_array.shape[0]
    
    angles_degrees = np.linspace(0, 180.0, side_length * 2 , endpoint=False)
    
    sinogram = radon_rect_gpu_processing.radon_transform(image_array, angles_degrees)
    
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), dpi=100)
    
    ax1.set_title(f"Eredeti kép - {image_name}", fontsize=12)
    ax1.imshow(image_array, cmap='gray', extent=[0, side_length, side_length, 0], origin='upper')
    ax1.tick_params(axis='both', labelsize=10)
    
    ax2.imshow(sinogram, cmap='gray', aspect='auto', extent=[0, sinogram.shape[1], 0, sinogram.shape[0]])
    ax2.tick_params(axis='both', labelsize=10)
    ax2.set_xlabel("Szögszám", fontsize=12)
    ax2.set_ylabel("Sugárszám", fontsize=12)
    
    filename_tif = f"results/{image_name}.tif"
    imsave(filename_tif, sinogram.astype(np.float32))
    
    plt.tight_layout()
    plt.show()
    
    
    end_time = time.time()
    print("Futás idő: {:.4f} másodperc".format(end_time - start_time))
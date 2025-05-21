import tomopy
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imsave
from PIL import Image


# Tomopy alapú rekonstrukció automatikus középpont meghatározással és az eredmény megjelenítésével.
def reconstruct(sinogram, image_name, image_title):
    image_width = sinogram.shape[1]
    
    sinogram = sinogram.swapaxes(0,1)
    data_stack = sinogram[:,np.newaxis, :]
    
    angles_degrees = np.linspace(0.0, 180.0, image_width, endpoint=False)
    angles_radians = np.deg2rad(angles_degrees)
    
    center = tomopy.find_center(data_stack, theta = angles_radians)
    print(f"Javasolt középpont: {center}")
    
    reconstruction = tomopy.recon(data_stack, angles_radians, center = center, 
                                  algorithm ='gridrec', filter_name = "cosine")  
    
    filename_png = f"results/{image_name}_{image_title}.png"
    filename_tif = f"results/{image_name}_{image_title}.tif"
    
    imsave(f"{filename_tif}", np.rot90(reconstruction[0], 2))
    
    plt.figure(figsize=(6, 6))
    plt.imshow(np.rot90(reconstruction[0], 2), cmap='gray')
    plt.title(f"{image_title}")
    plt.savefig(filename_png)
    plt.show()
    

# Betölti a megadott szinogramképet, átalakítja tömbbé és elindítja a rekonstrukciót.
def read_and_load_reconstruct(file_path, image_name, image_title):
    
    sinogram_img = Image.open(file_path)
    sinogram = np.array(sinogram_img, dtype=np.float64)
    reconstruct(sinogram, image_name, image_title)


if __name__ == '__main__':
    
    # Lehetőség van kiválasztani, melyik eljárás szinogramját szeretnénk rekonstruálni.
    use_tomopy = True
    use_rect = True
    use_circ = True
    
    image_name = "256c10"
    
    if use_tomopy:
        read_and_load_reconstruct(f'../radon_tomopy/results/{image_name}.tif', image_name, "TomoPy")
    
    if use_rect:
        read_and_load_reconstruct(f'../radon_rect/gpu/results/{image_name}.tif', image_name, "RECT")
    
    if use_circ:
        read_and_load_reconstruct(f'../radon_circ/gpu/results/{image_name}.tif', image_name, "CIRC")

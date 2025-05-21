import numpy as np
import matplotlib.pyplot as plt
import image_generation_gpu_processing as gpu_processing
from tifffile import imsave


if __name__ == '__main__':

    
    img_size = (256, 256)
    circle_num = 101
    
    min_radius = 42
    max_intensity = 1.2
    min_intesity = 0.1
    margin = 21
    
    is_float = True
    
    file_name = "256c101"
    
    image = gpu_processing.generate_image(img_size, file_name, circle_num, margin, min_radius, 
                                          min_intesity, max_intensity, is_float)

    
    fig, ax = plt.subplots()
    
    img = ax.imshow(image, cmap='gray', extent=[0, img_size[1], img_size[0], 0])
    ax.set_xticks(np.linspace(0, img_size[1], num=6))
    ax.set_yticks(np.linspace(0, img_size[0], num=6))
    ax.set_xticklabels([f"{t:.0f}" for t in ax.get_xticks()])
    ax.set_yticklabels([f"{t:.0f}" for t in ax.get_yticks()])
    cbar = fig.colorbar(img, ax=ax)
    ticks = np.linspace(image.min(), image.max(), num=8)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{t:.1f}" for t in ticks])
    #plt.grid()
    ax.set_title(f"Generált kép - {file_name}")
    
    filename_tif = f"{file_name}.tif"
    filename_png = f"results/{file_name}.png"
    
    imsave(f'../../test_images/{filename_tif}', image)

    plt.savefig(filename_png)
    
    plt.show()
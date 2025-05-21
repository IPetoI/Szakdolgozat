import time
import random
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imsave


# Körök kirajzolása egy megadott méretű képre, egy előre definiált paraméterlistát követve.
def create_image(data, img_size):
    image = np.zeros(img_size, dtype=np.float32)

    non_zero_pixel_count = 0
    max_intensity_val = 0
    
    for y in range(img_size[1]):
        for x in range(img_size[0]):
            for circle in data:
                center_x, center_y, radius, intensity = circle
                
                distance = np.sqrt(((x + 0.5) - center_x)**2 + ((y + 0.5) - center_y)**2)
                
                if distance < radius:
                    if image[y, x] == 0.0:
                        non_zero_pixel_count += 1
                        
                    image[y, x] += intensity
                    
                    if image[y, x] > max_intensity_val:
                        max_intensity_val = image[y, x]
    
    print(f'\nNem nulla intenzitású pixelek száma: {non_zero_pixel_count}. \
            \nIntenzitás eltérés maximum értéke: {max_intensity_val}.\n')
    
    return image
        

# A körök jellemzőit (x, y középpont, sugár és intenzitás) véletlenszerű lebegőpontos értékekkel generálja.
def generate_circles(num_circles, img_size, margin, min_radius, min_intesity, max_intensity, is_float):
    circles = []
    size = img_size[0]
    
    for _ in range(num_circles):
        max_radius = (size - margin * 2) // 2
        
        if is_float:
            radius = random.uniform(min_radius, max_radius)
            
            center_x = random.uniform(margin + radius, size - margin - radius)
            center_y = random.uniform(margin + radius, size - margin - radius)
            
            intensity = random.uniform(min_intesity, max_intensity)  
        else:
            radius = random.randint(min_radius, max_radius)
            
            center_x = random.randint(margin + radius, size - margin - radius)
            center_y = random.randint(margin + radius, size - margin - radius)
            
            intensity = random.randint(min_intesity, max_intensity)
        
        circles.append([center_x, center_y, radius, intensity])
    
    return circles



if __name__ == '__main__':

    start_time = time.time()
    
    
    img_size = (256, 256)
    num_circles  = 10
    
    min_radius = 21
    max_intensity = 10
    min_intesity = 2
    margin = 21
    
    is_float = True
    
    file_name = "256c10"
    
    circles = generate_circles(num_circles , img_size, margin, min_radius, min_intesity, max_intensity, is_float)
    
    image = create_image(circles, img_size)
    
    with open(f'../../txt_files/{file_name}.txt', 'w') as file:
        file.write(repr(circles))
    
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
    
    
    end_time = time.time()
    print("Futás idő: {:.4f} másodperc".format(end_time - start_time))

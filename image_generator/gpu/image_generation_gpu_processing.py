import math
import random
import numpy as np
from numba import cuda


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


# Körök kirajzolása egy megadott méretű képre, egy előre definiált paraméterlistát követve, gpu használatával.
@cuda.jit
def create_image(circles, img_size, image):
    x, y = cuda.grid(2)
    
    if x >= img_size[0] or y >= img_size[1]:
        return
    
    for circle in circles:
        center_x, center_y, radius, intensity = circle
        
        distance = math.sqrt(((x + 0.5) - center_x)**2 + ((y + 0.5) - center_y)**2)
        
        if distance < radius: 
            image[y, x] += intensity



def generate_image(img_size, file_name, circle_num, margin, min_radius, min_intesity, max_intensity, is_float):
    
    image = np.zeros(img_size, dtype=np.float32)

    threads_per_block = (16, 16)
    blocks_per_grid = (math.ceil(img_size[0] / threads_per_block[0]), 
                       math.ceil(img_size[1] / threads_per_block[1]))
        
    circles = generate_circles(circle_num, img_size, margin, min_radius, min_intesity, max_intensity, is_float)
    
    with open(f'../../txt_files/{file_name}.txt', 'w') as file:
        file.write(repr(circles))
        
    d_circles = cuda.to_device(circles)
    d_image = cuda.to_device(image)

    create_image[blocks_per_grid, threads_per_block](d_circles, img_size, d_image)
    
    d_image.copy_to_host(image)
    
    return image
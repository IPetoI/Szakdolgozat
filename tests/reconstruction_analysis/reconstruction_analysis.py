import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tifffile import imsave


# Hisztogram készítése a rekonstrukciók különbségtömbjei alapján, majd a teszteredmények kiírása.
def plot_histogram(difference_array, mse, image_title, image_name):    
    min_val = np.min(difference_array)
    max_val = np.max(difference_array)
    max_abs_error = np.max(np.abs(difference_array))
    mae = np.mean(np.abs(difference_array))
    
    plt.hist(difference_array.ravel(), bins=255, color='blue', alpha=1)
    plt.title(f"Eredeti kép - {image_title} rekonstrukció különbsége\n" +
              f"MaxAE: {max_abs_error:.6f}, Mae: {mae:.6f}, Mse: {mse:.6f}")
    plt.xlabel("Intenzitás különbség")
    plt.ylabel("Gyakoriság")
    
    filename_png = f"results/recon_diff_hist_original_img_{image_title}_{image_name}.png"
    plt.savefig(filename_png)
    
    plt.show()
    
    print(f"Legkisebb különbség értéke: {min_val}, \nLegnagyobb különbség értéke: {max_val}")
    print(f"MaxAE értéke: {max_abs_error}")
    print(f"Mae értéke: {mae}")
    print(f'Mse értéke: {mse}')


# A különbségtömb vizuális megjelenítése (plotolása).
def plot_difference(difference_array, image_title, image_name):
    plt.figure(figsize=(6, 6))
    plt.imshow(difference_array, cmap='gray')
    plt.title(f"Eredeti kép - {image_title} rekonstrukció különbsége")
    plt.axis('off')
    plt.show()
        
    filename_difference = f"results/difference_{image_name}-{image_title}.tif"
    imsave(filename_difference, difference_array.astype(np.float32))
    
    plt.show()
    
    
# A rekonstruált kép átméretezése az eredeti kép méretéhez igazítva.
def crop_image(target_size, difference_array, image_title):
    side_length = difference_array.shape[0]
    
    padding_left = (side_length - target_size) // 2
    padding_right = side_length - target_size - padding_left
    padding_top = (side_length - target_size) // 2
    padding_bottom = side_length - target_size - padding_top
    
    cropped_img = difference_array[
        padding_top : side_length - padding_bottom,
        padding_left : side_length - padding_right
        ]
    
    expanded_array = np.array(cropped_img, dtype=np.float64)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(expanded_array, cmap='gray')
    plt.title(f"Eredeti kép - {image_title} rekonstrukció különbsége")
    plt.axis('off')
    plt.show()
    
    
# A négyzetes hiba átlaga ( MSE ) kiszámítása.
def mean_squared_error(reconstructed_array, original_array):
    if original_array.shape != reconstructed_array.shape:
        raise ValueError("A két kép mérete nem egyezik meg!")
    
    mse = np.mean((original_array - reconstructed_array) ** 2)
    
    return mse


# A rekonstruált és eredeti kép közötti különbségtömb kiszámítása.
def difference(reconstructed_array, original_array):
    if original_array.shape != reconstructed_array.shape:
        raise ValueError("A két kép mérete nem egyezik meg!")
    
    difference = original_array - reconstructed_array
    
    return difference

    
# A képek átméretezése egységes méretre, ha eltérés van a méretek között.
def image_resizing(target_size, side_length):
    padding_left = (target_size - side_length) // 2
    padding_right = target_size - side_length - padding_left
    padding_top = (target_size - side_length) // 2
    padding_bottom = target_size - side_length - padding_top
    
    expanded_image = ImageOps.expand(original_img, border=(padding_left, padding_top, padding_right, 
                                                             padding_bottom), fill=0)
    expanded_array = np.array(expanded_image, dtype=np.float64)
    
    return expanded_array
    

# Rekonstruált kép betöltése, összehasonlítása az eredeti képpel, majd különbségek kiértékelése és megjelenítése.
def read_and_load_tests(file_path, image_name, image_title, original_array):
    reconstruct_image = Image.open(file_path)
    reconstruct_array = np.array(reconstruct_image, dtype=np.float64)
    
    expanded_array = image_resizing(reconstruct_array.shape[0], original_array.shape[0])
    
    mse = mean_squared_error(reconstruct_array, expanded_array)
    difference_array = difference(reconstruct_array, expanded_array)
    
    if crop:
        crop_image(original_array.shape[0], difference_array, image_title)
    
    plot_difference(difference_array, image_title, image_name)
    print(f"\nEredeti kép - {image_title} rekonstrukció különbsége:")
    plot_histogram(difference_array, mse, image_title, image_name)
    

if __name__ == "__main__":  
    
    # Lehetőség van kiválasztani, melyik eljárás rekonstrukcióját szeretnénk ellenőrizni.
    use_tomopy = True
    use_rect = True
    use_circ = True
    
    # Rekonstruált kép átméretezése eredeti méretre.
    crop = False
    
    image_name = "256c10"
    
    original_img = Image.open(f'../../test_images/{image_name}.tif')
    original_array = np.array(original_img, dtype=np.float64)
    
    
    if use_circ:
        read_and_load_tests(f'../../reconstruction_tomopy/results/{image_name}_circ.tif', image_name, "CIRC", original_array)
    
    if use_rect:
        read_and_load_tests(f'../../reconstruction_tomopy/results/{image_name}_rect.tif', image_name, "RECT", original_array)
        
    if use_tomopy:
        read_and_load_tests(f'../../reconstruction_tomopy/results/{image_name}_tomopy.tif', image_name, "tomopy", original_array)
        
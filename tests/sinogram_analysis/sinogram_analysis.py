import numpy as np
import matplotlib.pyplot as plt
from tifffile import imsave
from PIL import Image


# Hisztogram készítése a szinogramok különbségtömbje alapján, majd a teszteredmények kiírása.
def plot_histogram(difference_array, mse, image_title, image_name):    
    min_val = np.min(difference_array)
    max_val = np.max(difference_array)
    max_abs_error = np.max(np.abs(difference_array))
    mae = np.mean(np.abs(difference_array))
    
    plt.hist(difference_array.ravel(), bins=255, color='blue', alpha=1)
    plt.title(f"TomoPy - {image_title} szinogram különbsége\n" +
              f"MaxAE: {max_abs_error:.6f}, Mae: {mae:.6f}, Mse: {mse:.6f}")
    plt.xlabel("Intenzitás különbség")
    plt.ylabel("Gyakoriság")
    
    filename_png = f"results/sin_diff_hist_tomopy_{image_title}_{image_name}.png"
    plt.savefig(filename_png)
    
    plt.show()
    
    print(f"Legkisebb különbség értéke: {min_val} \nLegnagyobb különbség értéke: {max_val}")
    print(f"MaxAE értéke: {max_abs_error}")
    print(f"Mae értéke: {mae}")
    print(f'Mse értéke: {mse}')


# A különbségtömb vizuális megjelenítése (plotolása).
def plot_difference(difference_array, image_title, image_name):
    plt.figure(figsize=(6, 6))
    plt.imshow(difference_array, cmap='gray')
    plt.title(f"TomoPy - {image_title} Különbsége")
    plt.axis('off')
    plt.show()
        
    filename_difference = f"results/difference_{image_name}-{image_title}.tif"
    imsave(filename_difference, difference_array.astype(np.float32))


# A négyzetes hiba átlaga ( MSE ) kiszámítása.
def mean_squared_error(reconstructed_array, tomopy_array):
    if tomopy_array.shape != reconstructed_array.shape:
        raise ValueError("A két kép mérete nem egyezik meg!")
    
    mse = np.mean((tomopy_array - reconstructed_array) ** 2)
    
    return mse


# A rekonstruált és eredeti kép közötti különbségtömb kiszámítása.
def difference(reconstructed_array, tomopy_array):
    if tomopy_array.shape != reconstructed_array.shape:
        raise ValueError("A két kép mérete nem egyezik meg!")
    
    difference = tomopy_array - reconstructed_array
    
    return difference


# Szinogram betöltése, Tomopy szinogrammal való összehasonlítása, majd különbségek kiértékelése és megjelenítése.
def read_and_load_tests(file_path, image_name, image_title, tomopy_sinogram_array):
    sinogram = Image.open(file_path)
    sinogram_array = np.array(sinogram, dtype=np.float64)
    
    mse = mean_squared_error(sinogram_array, tomopy_sinogram_array)
    difference_array = difference(sinogram_array, tomopy_sinogram_array)

    plot_difference(difference_array, image_title, image_name)
    print(f"\nTomoPy - {image_title} szinogram különbsége:")
    plot_histogram(difference_array, mse, image_title, image_name)



if __name__ == "__main__":   
    
    # Lehetőség van kiválasztani, melyik eljárás rekonstrukcióját szeretnénk ellenőrizni.
    use_rect = True
    use_circ = True
    
    image_name = "256c10"
    
    tomopy_sinogram = Image.open(f'../../radon_tomopy/results/{image_name}.tif')
    tomopy_sinogram_array = np.array(tomopy_sinogram, dtype=np.float64)  
    
    if use_rect:
        read_and_load_tests(f'../../radon_rect/gpu/results/{image_name}.tif', image_name, "RECT", tomopy_sinogram_array)
    
    if use_circ:
        read_and_load_tests(f'../../radon_circ/gpu/results/{image_name}.tif', image_name, "CIRC", tomopy_sinogram_array)

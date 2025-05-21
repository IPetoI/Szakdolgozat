import matplotlib.pyplot as plt
import numpy as np


# Vizuális szemléltetése a Radon-transzformáció folyamatának.
def plot_circ_with_line(line_center, line_direction, circle_center, circle_radius, side_length, background_image):
    plt.figure(figsize=(6, 6))
    
    # Háttérkép megjelenítése.
    plt.imshow(background_image, extent=[0, side_length, side_length, 0], origin='upper', 
           cmap='gray', alpha=1)

    # Egyenes kirajzolása.
    t_values = np.linspace(-side_length * np.sqrt(2), side_length * np.sqrt(2))
    line_points = np.array([(line_center + t * line_direction) for t in t_values])
    plt.plot(line_points[:, 1], line_points[:, 0], 'r-', lw=1)
    
    # Kör kirajzolása.
    circle = plt.Circle((circle_center[1], circle_center[0]), circle_radius, color='blue', fill=False, lw=1.5)
    plt.gca().add_artist(circle)
    
    plt.axhline(y=side_length / 2, color='gray', linestyle='--')
    plt.axvline(x=side_length / 2, color='gray', linestyle='--')
    plt.xlim(0, side_length)
    plt.ylim(0, side_length)
    #plt.grid(True)

    plt.show()


# A sugár és a circ() bázisfüggvény által meghatározott kör metszéspontjainak kiszámítása 
#       másodfokú egyenlet alapján.
def calculate_circ_line_intersections(circle_center, circle_radius, angle_rad_line, line_center, intersection_points, 
                                        plot, side_length, image_array):
    
    line_direction = np.array([[np.cos(angle_rad_line)], [np.sin(angle_rad_line)]], dtype=np.float64)
    
    dx, dy = line_direction.flatten()
    fx, fy = line_center.flatten() - circle_center.flatten()

    # Másodfokú egyenlet együtthatóinak számítása.
    a = dx**2 + dy**2
    b = 2 * (fx * dx + fy * dy)
    c = fx**2 + fy**2 - circle_radius**2

    # Gyök alatti rész ( diszkrimináns ) számítása.
    discriminant = b**2 - 4 * a * c
    
    if plot:    
        plot_circ_with_line(line_center, line_direction, circle_center, circle_radius, side_length, image_array)
    
    # Ha a diszkrimináns negatív értéket vesz fel, abban az esetben nincs metszéspont, ha nullát akkor viszont csak ériti.
    if discriminant <= 0:
        return 0
    
    # Lehetséges metszéspontok kiszámítása.
    sqrt_discriminant = np.sqrt(discriminant)
    
    t1 = (-b - sqrt_discriminant) / (2 * a)
    t2 = (-b + sqrt_discriminant) / (2 * a)
    
    num_intersections = 0
    
    for t in [t1, t2]:
        V = line_center + t * line_direction
        intersection_points.append(V)
        num_intersections += 1

    return num_intersections


# A sugár körön belüli maximális távolságának meghatározása a metszéspontok alapján.
def calculate_max_distance(intersection_points):
    
    dx = intersection_points[0][0] - intersection_points[1][0]
    dy = intersection_points[0][1] - intersection_points[1][1]
    
    return np.sqrt(dx ** 2 + dy ** 2)


# Radon-transzformáció circ() bázisfüggvénnyel.
def radon_transform(data, angles_degrees, side_length, plot, image_array):
    divisor = np.floor((side_length * np.sqrt(2) - side_length + 4) / 2)
    num_columns = (divisor * 2 + side_length)

    sinogram = np.zeros((int(num_columns), len(angles_degrees)), dtype=np.float64)
    
    # A sugár szögek szerinti forgatása.
    for angle_idx, angle in enumerate(angles_degrees):
        angle_rad = np.radians(angle)
        
        if plot:
            print("Szög: ",angle)
        
        # A sugár léptetése az x tengelyen.
        for offset_idx, line_x_offset in enumerate(np.arange(0.5 - divisor, side_length + divisor)):
            # A sugár koordinátáinak számítása a képközéppont, forgatás és eltolás alapján.
            line_center = np.array([
                [side_length / 2 - (side_length / 2 * (np.sin(angle_rad))) + (line_x_offset * (np.sin(angle_rad)))],
                [side_length / 2 + (side_length / 2 * (np.cos(angle_rad))) - (line_x_offset * (np.cos(angle_rad)))]], 
                dtype=np.float64)
            
            line_integral = 0
            
            # Körök ellenőrzése és feldolgozása.
            for circle in data:
                center_x, center_y, circle_radius, intensity = circle
                circle_center = np.array([[center_y], [center_x]], dtype=np.float64)
                
                intersection_points = []
                
                num_intersections = calculate_circ_line_intersections(circle_center, circle_radius, angle_rad, 
                                                    line_center, intersection_points, plot, side_length, image_array)
                
                distance = calculate_max_distance(intersection_points) if num_intersections == 2 else 0
                
                line_integral += intensity * distance
            
            sinogram[offset_idx, angle_idx] = line_integral
        
    return sinogram
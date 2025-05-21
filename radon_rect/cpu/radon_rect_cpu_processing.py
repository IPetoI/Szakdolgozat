import matplotlib.pyplot as plt
import numpy as np


# Vizuális szemléltetése a Radon-transzformáció folyamatának.
def plot_rect_with_lines(line_center, line_direction, pixel_corner, pixel_edge_direction, pixel_y, pixel_x, side_length, 
                         background_image):
    plt.figure(figsize=(6, 6))
    
    # Háttérkép megjelenítése.
    plt.imshow(background_image, extent=[0, side_length, side_length, 0], origin='upper', 
                   cmap='gray', alpha=1)

    # Egyenes kirajzolása.
    t_values = np.linspace(-side_length  * np.sqrt(2), side_length  * np.sqrt(2))
    line_points = np.array([(line_center + t * line_direction) for t in t_values])
    plt.plot(line_points[:, 1], line_points[:, 0], 'r-', lw=1)
    
    # Adott pixel él egyenesének kirajzolása.
    t_values = np.linspace(-side_length * np.sqrt(2), side_length * np.sqrt(2))
    line_points = np.array([(pixel_corner + t * pixel_edge_direction) for t in t_values])
    plt.plot(line_points[:, 1], line_points[:, 0], 'b-', lw=1)
    
    # Alsó-felső-bal-jobb pixel oldal kirajzolása.
    plt.plot([pixel_x, pixel_x], [pixel_y, pixel_y + 1], 'g-', lw=1.5)
    plt.plot([pixel_x + 1, pixel_x + 1], [pixel_y, pixel_y + 1], 'g-', lw=1.5)
    plt.plot([pixel_x, pixel_x + 1], [pixel_y, pixel_y], 'g-', lw=1.5)
    plt.plot([pixel_x, pixel_x + 1], [pixel_y + 1, pixel_y + 1], 'g-', lw=1.5)

    plt.axhline(y=side_length / 2, color='gray', linestyle='--')
    plt.axvline(x=side_length / 2, color='gray', linestyle='--')
    plt.xlim(0, side_length)
    plt.ylim(0, side_length)
    #plt.grid(True)

    plt.show()
    

# A sugár irány és normálvektorának számítása.
def calculate_line_normal_vektor(angle_rad_line, rotation_90_matrix):
    line_direction = np.array([[np.cos(angle_rad_line)], [np.sin(angle_rad_line)]], dtype=np.float64) 
    line_normal = np.dot(rotation_90_matrix, line_direction)
    
    return line_direction, line_normal


# Bresenham vonal algoritmus használata a sugár körüli pixelek kiszűrésére.
def find_nearby_pixels_along_line(line_direction, line_center, side_length): 
    t_range = side_length / 2 + side_length / 4
    
    t_values = np.array([-int(t_range), int(t_range)])
    
    line_segment_ends = np.array([(line_center + t * line_direction) for t in t_values])
    
    x1, y1 = np.round(line_segment_ends[0]).astype(int).flatten()
    x2, y2 = np.round(line_segment_ends[1]).astype(int).flatten()
    
    bresenham_pixels = set()
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    
    err = dx - dy
    
    while True:
        bresenham_pixels.add((x1, y1))

        if x1 == x2 and y1 == y2:
            break

        e2 = err * 2

        if e2 > -dy:
            err -= dy
            x1 += sx

        if e2 < dx:
            err += dx
            y1 += sy
            
    candidate_pixels = set()
    
    # Egyenes körüli pixelek, és ismétlések kiszűrése.
    for x, y in bresenham_pixels:
        for ny in range(-2, 2):
            for nx in range(-2, 2):
                if 0 <= x + nx < side_length and 0 <= y + ny < side_length:
                    candidate_pixels.add((x + nx, y + ny))
   
    return sorted(candidate_pixels)


# A sugár és a rect() bázisfüggvény által meghatározott pixel élek metszéspontjainak kiszámítása 
#       lineáris egyenletrendszer alapján.
def calculate_rect_line_intersections(angle_rad_pixel, pixel_corner, line_center, pixel, intersection_points, 
                                       line_normal, rotation_90_matrix, plot, side_length, intensity_array, line_direction):

    pixel_edge_direction = np.array([[np.cos(angle_rad_pixel)], [np.sin(angle_rad_pixel)]], dtype=np.float64)
    
    pixel_normal = np.dot(rotation_90_matrix, pixel_edge_direction)
    
    # Lineáris egyenletrendszer összeállítása ( Ax = b )
    A = np.concatenate((line_normal.T, pixel_normal.T), axis=0)
    
    b = np.array([np.dot(line_normal.T, line_center).item(), np.dot(pixel_normal.T, pixel_corner).item()])
    
    if plot:
        plot_rect_with_lines(line_center, line_direction, pixel_corner, pixel_edge_direction, pixel[0], pixel[1], 
                             side_length, intensity_array)
        
    epsilon = 1e-6
    
    # Ha a sugár és a pixel egyenese párhuzamos, vagyis nincs metszéspont,
    #       akkor az A mátrix determinánsa nullához közeli értéket vesz fel.
    if abs(np.linalg.det(A)) < epsilon:
        if plot:
            print("A két vonal párhuzamos, nincs metszéspont.")
        return
    
    V = np.linalg.solve(A, b).reshape(2, 1)
    
    # Ellenőrzés hogy a pixel határokon belül legyen a talált metszéspont.
    if np.isclose(angle_rad_pixel, np.pi / 2):  # függőleges él.
        if (pixel[1] - epsilon) <= V[1] <= (pixel[1] + 1 + epsilon):
            intersection_points.append(V)
    
    elif np.isclose(angle_rad_pixel, np.pi):  # vízszintes él.
        if (pixel[0] - epsilon) <= V[0] <= (pixel[0] + 1 + epsilon):
            intersection_points.append(V)


# A sugár pixelen belüli maximális távolságának meghatározása a metszéspontok alapján.
def calculate_max_distance(intersection_points):
    max_distance = 0
    
    for i in range(len(intersection_points)):
        for j in range(i+1, len(intersection_points)):
            distance = np.linalg.norm(intersection_points[i] - intersection_points[j])
            max_distance = max(max_distance, distance)
                
    return max_distance


# Radon-transzformáció rect() bázisfüggvénnyel.
def radon_transform(intensity_array, angles_degrees, plot):
    side_length = intensity_array.shape[0]

    rotation_90_matrix = np.array([[np.cos(np.pi / 2), -np.sin(np.pi / 2)], 
                     [np.sin(np.pi / 2), np.cos(np.pi / 2)]], dtype=np.float64)
    
    divisor = np.floor((side_length * np.sqrt(2) - side_length + 4) / 2)
    max_columns = (divisor * 2 + side_length)
    
    sinogram = np.zeros((int(max_columns), len(angles_degrees)), dtype=np.float64)
    
    # A sugár szögek szerinti forgatása.
    for angle_idx, angle in enumerate(angles_degrees):
        angle_rad = np.radians(angle)
        line_direction, line_normal = calculate_line_normal_vektor(angle_rad, rotation_90_matrix)
        
        if plot:
            print("Szög: ", angle)
        
        # A sugár léptetése az x tengelyen.
        for offset_idx, line_x_offset in enumerate(np.arange(0.5 - divisor, side_length + divisor)):
            # A sugár koordinátáinak számítása a képközéppont, forgatás és eltolás alapján.
            line_center = np.array([
                [side_length / 2 - (side_length / 2 * (np.sin(angle_rad))) + (line_x_offset * (np.sin(angle_rad)))],
                [side_length / 2 + (side_length / 2 * (np.cos(angle_rad))) - (line_x_offset * (np.cos(angle_rad)))]], 
                dtype=np.float64)
            
            line_integral = 0
            
            candidate_pixels = find_nearby_pixels_along_line(line_direction, line_center, side_length)

            # A sugár közelében lévő pixelek feldolgozása.
            for x, y in candidate_pixels:
                
                # Nulla intenzitású pixelek kiszűrése (gyorsítás céljából).
                if intensity_array[x][y] == 0.0:
                    continue

                pixel_corner = np.array([[x], [y]], dtype=np.float64)
                
                pixel = np.array([[x], [y]], dtype=np.int64)
                
                intersection_points = []
                
                # A pixel 4 oldalának külön vizsgálata a metszéspontok meghatározásához.
                for pixel_edge in range(0, 4): 
                    angle_rad_pixel = np.pi if pixel_edge >= 2 else (np.pi / 2)
                    
                    if pixel_edge == 1:   # fent
                        pixel_corner[0] += 1
                        
                    elif pixel_edge == 3: # jobbra
                        pixel_corner[1] += 1
                        
                    calculate_rect_line_intersections(angle_rad_pixel, pixel_corner, line_center, pixel, intersection_points, 
                                             line_normal, rotation_90_matrix, plot, side_length, intensity_array, line_direction)
                
                num_intersections = len(intersection_points)

                distance = calculate_max_distance(intersection_points) if 2 <= num_intersections <= 4 else 0
                
                line_integral += intensity_array[x][y] * distance
        
            sinogram[offset_idx, angle_idx] = line_integral

    return sinogram
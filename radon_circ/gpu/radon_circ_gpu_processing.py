import math
import numpy as np
from numba import cuda, float64


# A sugár és a circ() bázisfüggvény által meghatározott kör metszéspontjainak kiszámítása 
#       másodfokú egyenlet alapján.
@cuda.jit(device=True)
def calculate_circ_line_intersections(circle_center, circle_r, angle_rad_line, line_center, intersection_points):
    line_direction = cuda.local.array(2, dtype=float64)
    
    line_direction[0] = math.cos(angle_rad_line)
    line_direction[1] = math.sin(angle_rad_line)
    
    dx = line_direction[0]
    dy = line_direction[1]
    
    fx = line_center[0] - circle_center[0]
    fy = line_center[1] - circle_center[1]

    a = dx**2 + dy**2
    b = 2 * (fx * dx + fy * dy)
    c = fx**2 + fy**2 - circle_r**2

    discriminant = b**2 - 4 * a * c

    # Ha a diszkrimináns negatív értéket vesz fel, abban az esetben nincs metszéspont, ha nullát akkor viszont csak ériti.
    if discriminant <= 0:
        return 0
    
    sqrt_discriminant = math.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2 * a)
    t2 = (-b + sqrt_discriminant) / (2 * a)
    
    intersection_count = 0
    
    for t in [t1, t2]:
        
        Vx = line_center[0] + t * line_direction[0]
        Vy = line_center[1] + t * line_direction[1]
        
        intersection_points[intersection_count, 0] = Vx
        intersection_points[intersection_count, 1] = Vy
        intersection_count += 1
    
    return intersection_count


# A sugár körön belüli maximális távolságának meghatározása a metszéspontok alapján.
@cuda.jit(device=True)
def calculate_max_distance(intersection_points):

    dx = intersection_points[0, 0] - intersection_points[1, 0]
    dy = intersection_points[0, 1] - intersection_points[1, 1]
    
    return math.sqrt(dx * dx + dy * dy)


# Radon-transzformáció circ() bázisfüggvénnyel.
@cuda.jit
def calculate_radon_transform_circ(data, angles_degrees, sinogram, side_length, divisor):
    angle_idx, offset = cuda.grid(2)
    
    if angle_idx >= angles_degrees.shape[0] or offset >= (side_length + divisor * 2):
        return

    angle = angles_degrees[angle_idx]
    angle_rad = math.radians(angle)

    line_x_offset = offset + 0.5 - divisor
    line_center = cuda.local.array(2, dtype=float64)
    
    half_side_length = side_length / 2
    sin_angle_rad = math.sin(angle_rad)
    cos_angle_rad = math.cos(angle_rad)
    
    # A sugár koordinátáinak számítása a képközéppont, forgatás és eltolás alapján.
    line_center[0] = half_side_length - (half_side_length * sin_angle_rad) + (line_x_offset * sin_angle_rad)
    line_center[1] = half_side_length + (half_side_length * cos_angle_rad) - (line_x_offset * cos_angle_rad)

    projection_sum = 0
    
    # Körök ellenőrzése és feldolgozása.
    for circle in data:
        center_x = circle[0]
        center_y = circle[1]
        radius = circle[2]
        intensity = circle[3]
        
        circle_center = cuda.local.array(2, dtype=float64)
        
        circle_center[0] = center_y
        circle_center[1] = center_x
        
        intersection_points = cuda.local.array((2, 2), dtype=float64)
        for i in range(2):
            intersection_points[i, 0] = math.nan
            intersection_points[i, 1] = math.nan
        
        num_intersections = calculate_circ_line_intersections(circle_center, radius, angle_rad, line_center, intersection_points)
        
        if num_intersections == 2:
            distance = calculate_max_distance(intersection_points)
        else:
            distance = 0
        
        projection_sum += intensity * distance
        
    sinogram[offset, angle_idx] = projection_sum



# A Radon-transzformáció GPU-alapú futtatásához szükséges adatelőkészítés és kernelhívás.
def radon_transform(data, angles_degrees, side_length):
    divisor = np.floor((side_length * np.sqrt(2) - side_length + 4) / 2)
    num_columns = (divisor * 2 + side_length)

    sinogram = np.zeros((int(num_columns), len(angles_degrees)), dtype=np.float64)

    d_data = cuda.to_device(data)
    d_angles_degrees = cuda.to_device(np.array(angles_degrees, dtype=np.float64))
    d_sinogram = cuda.to_device(sinogram)

    threads_per_block = (16, 16)
    blocks_per_grid = (math.ceil(len(angles_degrees) / threads_per_block[0]), 
                       math.ceil(num_columns / threads_per_block[1]))

    calculate_radon_transform_circ[blocks_per_grid, threads_per_block](d_data, d_angles_degrees, d_sinogram, 
                                                          side_length, divisor)

    d_sinogram.copy_to_host(sinogram)
    
    return sinogram

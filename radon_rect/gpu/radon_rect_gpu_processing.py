import math
import numpy as np
from numba import cuda, float64, int32


# A sugár irány és normálvektorának számítása.
@cuda.jit(device=True)
def calculate_line_normal_vector(angle_rad_line, line_direction, line_normal):
    R_90 = cuda.local.array((2, 2), dtype=float64)
    
    R_90[0, 0] = math.cos(math.pi / 2)
    R_90[0, 1] = -math.sin(math.pi / 2)
    R_90[1, 0] = math.sin(math.pi / 2)
    R_90[1, 1] = math.cos(math.pi / 2)
    
    line_direction[0] = math.cos(angle_rad_line)
    line_direction[1] = math.sin(angle_rad_line)
    
    line_normal[0] = R_90[0, 0] * line_direction[0] + R_90[0, 1] * line_direction[1]
    line_normal[1] = R_90[1, 0] * line_direction[0] + R_90[1, 1] * line_direction[1]


# Bresenham vonal algoritmus használata a sugár körüli pixelek kiszűrésére.
@cuda.jit(device=True)
def find_nearby_pixels_along_line(line_direction, line_center, side_length, candidate_pixels):
    t_range = side_length / 2 + side_length / 4
    
    t_values = cuda.local.array(2, dtype=int32)
    
    t_values[0] = -int(t_range)
    t_values[1] = int(t_range)
    
    line_segment_ends = cuda.local.array((2, 2), dtype=float64)
    
    for i, t in enumerate(t_values):
        line_segment_ends[i, 0] = line_center[0] + t * line_direction[0]
        line_segment_ends[i, 1] = line_center[1] + t * line_direction[1]
    
    x1 = int(round(line_segment_ends[0, 0]))
    y1 = int(round(line_segment_ends[0, 1]))
    
    x2 = int(round(line_segment_ends[1, 0]))
    y2 = int(round(line_segment_ends[1, 1]))
    
    bresenham_pixels = cuda.local.array((600, 2), dtype=int32)
    
    pixel_count = 0
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    
    err = dx - dy
    
    while True:
        if pixel_count < 600:
            bresenham_pixels[pixel_count, 0] = x1
            bresenham_pixels[pixel_count, 1] = y1
            pixel_count += 1

        if x1 == x2 and y1 == y2:
            break

        e2 = err * 2

        if e2 > -dy:
            err -= dy
            x1 += sx

        if e2 < dx:
            err += dx
            y1 += sy
            
    num_candidates = 0
    
    visited_pixels = cuda.local.array((2100, 2), dtype=int32)
    
    # Ismétlések kiszűrése.
    for i in range(pixel_count):
        x = bresenham_pixels[i, 0]
        y = bresenham_pixels[i, 1]
        for nx in range(-2, 2):
            for ny in range(-2, 2):
                if 0 <= x + nx < side_length and 0 <= y + ny < side_length:
                    
                    is_duplicate = False
                    for idx in range(num_candidates):
                        if visited_pixels[idx, 0] == x + nx and visited_pixels[idx, 1] == y + ny:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate and num_candidates < 2100:
                        candidate_pixels[num_candidates, 0] = x + nx
                        candidate_pixels[num_candidates, 1] = y + ny
                        
                        visited_pixels[num_candidates, 0] = x + nx
                        visited_pixels[num_candidates, 1] = y + ny
                        num_candidates += 1


# A sugár és a rect() bázisfüggvény által meghatározott pixel élek metszéspontjainak kiszámítása 
#       lineáris egyenletrendszer alapján.
@cuda.jit(device=True)
def calculate_rect_line_intersections(angle_rad_pixel, pixel_corner, line_center, pixel, intersection_points, line_normal,
                                       pixel_edge, angle):
    
    R_90 = cuda.local.array((2, 2), dtype=float64)
    
    R_90[0, 0] = math.cos(math.pi / 2)
    R_90[0, 1] = -math.sin(math.pi / 2)
    R_90[1, 0] = math.sin(math.pi / 2)
    R_90[1, 1] = math.cos(math.pi / 2)

    pixel_r = cuda.local.array(2, dtype=float64)
    
    pixel_r[0] = math.cos(angle_rad_pixel)
    pixel_r[1] = math.sin(angle_rad_pixel)

    pixel_n = cuda.local.array(2, dtype=float64)
    
    pixel_n[0] = R_90[0, 0] * pixel_r[0] + R_90[0, 1] * pixel_r[1]
    pixel_n[1] = R_90[1, 0] * pixel_r[0] + R_90[1, 1] * pixel_r[1]

    A = cuda.local.array((2, 2), dtype=float64)
    
    A[0, 0] = line_normal[0]
    A[0, 1] = line_normal[1]
    A[1, 0] = pixel_n[0]
    A[1, 1] = pixel_n[1]

    b = cuda.local.array(2, dtype=float64)
    
    b[0] = line_normal[0] * line_center[0] + line_normal[1] * line_center[1]
    b[1] = pixel_n[0] * pixel_corner[0] + pixel_n[1] * pixel_corner[1]

    epsilon = 1e-6

    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]

    # Ha a sugár és a pixel egyenese párhuzamos, vagyis nincs metszéspont,
    #       akkor az A mátrix determinánsa nullához közeli értéket vesz fel.
    if abs(det) < epsilon:
        return

    inv_det = 1.0 / det
    
    A_inv = cuda.local.array((2, 2), dtype=float64)
    
    A_inv[0, 0] = A[1, 1] * inv_det
    A_inv[0, 1] = -A[0, 1] * inv_det
    A_inv[1, 0] = -A[1, 0] * inv_det
    A_inv[1, 1] = A[0, 0] * inv_det

    V = cuda.local.array(2, dtype=float64)
    
    V[0] = A_inv[0, 0] * b[0] + A_inv[0, 1] * b[1]
    V[1] = A_inv[1, 0] * b[0] + A_inv[1, 1] * b[1]

    # Mivel a sugár és a pixel egyenesének végtelen hossza miatt a metszéspont 
    #       nem feltétlenül a pixel határain belül van, így ezt ellenőrizni kell.
    if abs(angle_rad_pixel - math.pi / 2) < epsilon:
        if pixel[1] - epsilon <= V[1] <= pixel[1] + 1 + epsilon:
            intersection_points[pixel_edge, 0] = V[0]
            intersection_points[pixel_edge, 1] = V[1]
            
    elif abs(angle_rad_pixel - math.pi) < epsilon:
        if pixel[0] - epsilon <= V[0] <= pixel[0] + 1 + epsilon:
            intersection_points[pixel_edge, 0] = V[0]
            intersection_points[pixel_edge, 1] = V[1]


# A sugár pixelen belüli maximális távolságának meghatározása a metszéspontok alapján.
@cuda.jit(device=True)
def calculate_max_distance(intersection_points):
    max_distance = 0.0
    
    for i in range(4):
        if intersection_points[i, 0] == 0 and intersection_points[i, 1] == 0:
            continue
        
        for j in range(i + 1, 4):
            if intersection_points[j, 0] == 0 and intersection_points[j, 1] == 0:
                continue
            
            dx = intersection_points[i, 0] - intersection_points[j, 0]
            dy = intersection_points[i, 1] - intersection_points[j, 1]
            
            distance = math.sqrt(dx * dx + dy * dy)
            max_distance = max(max_distance, distance)
            
    return max_distance


# GPU-alapú Radon-transzformáció rect() bázisfüggvénnyel.
@cuda.jit
def calculate_radon_transform_rect(intensity_array, angles_degrees, sinogram, side_length, divisor):
    angle_idx, offset = cuda.grid(2)
    
    if angle_idx >= angles_degrees.shape[0] or offset >= (side_length + divisor * 2):
        return

    angle = angles_degrees[angle_idx]
    angle_rad_line = math.radians(angle)
    
    line_direction = cuda.local.array(2, dtype=float64)
    line_normal = cuda.local.array(2, dtype=float64)
    
    calculate_line_normal_vector(angle_rad_line, line_direction, line_normal)

    line_x_offset = offset + 0.5 - divisor
    
    pixel = cuda.local.array(2, dtype=float64)
    line_center = cuda.local.array(2, dtype=float64)
    
    half_side_length = side_length / 2
    sin_angle_rad_line = math.sin(angle_rad_line)
    cos_angle_rad_line = math.cos(angle_rad_line)
    
    # A sugár koordinátáinak számítása a képközéppont, forgatás és eltolás alapján.
    line_center[0] = half_side_length - (half_side_length * sin_angle_rad_line) + (line_x_offset * sin_angle_rad_line)
    line_center[1] = half_side_length + (half_side_length * cos_angle_rad_line) - (line_x_offset * cos_angle_rad_line)
    
    line_integral = 0.0
    
    candidate_pixels = cuda.local.array((2100, 2), dtype=int32)

    find_nearby_pixels_along_line(line_direction, line_center, side_length, candidate_pixels)
    
    # A sugár közelében lévő pixelek feldolgozása.
    for current_pixel in candidate_pixels:
        x = int(current_pixel[0])
        y = int(current_pixel[1])
        
        # Nulla intenzitású pixelek kiszűrése (gyorsítás céljából).
        if intensity_array[x, y] != 0.0:
            
            pixel[0] = x
            pixel[1] = y
                
            intersection_points = cuda.local.array((4, 2), dtype=float64)
            
            for i in range(4):
                intersection_points[i, 0] = math.nan
                intersection_points[i, 1] = math.nan

            # A pixel 4 oldalának külön vizsgálata a metszéspontok meghatározásához.
            for pixel_edge in range(4):
                angle_rad_pixel = math.pi if pixel_edge >= 2 else (math.pi / 2)
                
                pixel_corner = cuda.local.array(2, dtype=float64)
                
                pixel_corner[0] = x + (1 if pixel_edge == 1 else 0)
                pixel_corner[1] = y + (1 if pixel_edge == 3 else 0)

                calculate_rect_line_intersections(angle_rad_pixel, pixel_corner, line_center, pixel, intersection_points, 
                                   line_normal, pixel_edge, angle)

            num_intersections = 0
            
            for i in range(4):
                if not math.isnan(intersection_points[i, 0]) and not math.isnan(intersection_points[i, 1]):
                    num_intersections += 1

            if 2 <= num_intersections <= 4:
                distance = calculate_max_distance(intersection_points)
            else:
                distance = 0.0

            line_integral += intensity_array[x, y] * distance
        
    sinogram[offset, angle_idx] = line_integral
    
    
    
# A Radon-transzformáció GPU-alapú futtatásához szükséges adatelőkészítés és kernelhívás.
def radon_transform(intensity_array, angles_degrees):
    side_length = intensity_array.shape[0]
    divisor = np.floor((side_length * np.sqrt(2) - side_length + 4) / 2)
    num_columns = (divisor * 2 + side_length)

    sinogram = np.zeros((int(num_columns), len(angles_degrees)), dtype=np.float64)

    d_intensity_array = cuda.to_device(intensity_array)
    d_angles_degrees = cuda.to_device(np.array(angles_degrees, dtype=np.float64))
    d_sinogram = cuda.to_device(sinogram)

    threads_per_block = (16, 16)
    blocks_per_grid = (math.ceil(len(angles_degrees) / threads_per_block[0]), 
                       math.ceil(num_columns / threads_per_block[1]))

    calculate_radon_transform_rect[blocks_per_grid, threads_per_block](d_intensity_array, d_angles_degrees, 
                                                          d_sinogram, side_length, divisor)

    d_sinogram.copy_to_host(sinogram)
    
    return sinogram
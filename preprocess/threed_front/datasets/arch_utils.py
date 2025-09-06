# 
# Modified from: 
#   https://github.com/QiuhongAnnaWei/LEGO-Net/blob/main/data/preprocess_TDFront.py
# 

import numpy as np
import cv2 as cv
from scipy.interpolate import interp1d


def process_arch_iterative_closest_point(scene_data, room_side, key_prefix="floor_plan"):
    """ Returns ordered floorplan corners w.r.t. "arch_centroid": numpy array of shape [numpt, 2]
        scene_data: pre-processed scene data (loaded from boxes.npz)
        room_side: room_side parameter used to render room_layout in scene_data (can be an approximate)
        key_prefix: key prefix of the architecture element to process
    """    
    ## Source: contour points found on room_layout mask
    room_layout = np.squeeze(scene_data["room_layout"]*255 if scene_data["room_layout"].max() <= 3 else scene_data["room_layout"])
    kernel = np.ones((3,3), np.uint8)
    # Find thin lines (areas that would be removed by erosion)
    eroded = cv.erode(room_layout, kernel, iterations=1)
    thin_lines = room_layout - eroded
    # Perform your processing on the eroded image
    processed = eroded  # Replace with your actual processing
    # Add the thin lines back
    room_layout = processed + thin_lines
    all_contours, _ = cv.findContours(room_layout, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    floor_plan_centroid_2d = np.array([scene_data["floor_plan_centroid"][0], scene_data["floor_plan_centroid"][2]])
    
    if key_prefix == "floor_plan":
        # For floor plan, select largest contour
        contour = all_contours[0] if len(all_contours)==1 else max(all_contours, key=cv.contourArea)
    else:
        # For doors/windows, select contour closest to vertices
        vertices = scene_data[f"{key_prefix}_vertices"]
        # Project vertices to 2D
        vertices_2d = vertices[:, [0, 2]]  # Take x,z coordinates
        vertices_centroid = np.mean(vertices_2d, axis=0)
        vertices_centroid = vertices_centroid - floor_plan_centroid_2d
        
        # Find contour with centroid closest to vertices centroid
        min_dist = float('inf')
        closest_contour = None
        for cont in all_contours:
            cont_centroid = np.mean(cont.squeeze(), axis=0)
            # Convert contour coordinates to world coordinates for comparison
            cont_centroid = cont_centroid / (room_layout.shape[0]) * (room_side*2) - room_side
            dist = np.linalg.norm(cont_centroid - vertices_centroid)
            if dist < min_dist:
                min_dist = dist
                closest_contour = cont
        contour = closest_contour
    
    contour = np.squeeze(contour)  # (numcontourpt,1,2) -> (numcontourpt,2)
    try:
        contour = contour / (room_layout.shape[0]) * (room_side*2) - room_side  # Scale to world coordinates
    except:
        breakpoint()
    
    # Center contour points only for windows and doors
    if key_prefix in ["window", "door"]:
        contour_center = np.mean(contour, axis=0)
        contour = contour - contour_center

    ## Target: ATISS's generated points, extracted from 3DFRONT mesh objects in json
    center_2d = np.array([scene_data[f"{key_prefix}_centroid"][0], scene_data[f"{key_prefix}_centroid"][2]])
    vertices = scene_data[f"{key_prefix}_vertices"]  # Get all vertices
    
    # For windows and doors, get vertices from lowest plane
    if key_prefix in ["window", "door"]:
        min_y = vertices[:, 1].min()  # Find lowest y value
        vertices = vertices[vertices[:, 1] == min_y]  # Keep only vertices in lowest plane
        corners = np.unique(vertices, axis=0)  # Get unique vertices
    else:
        corners = np.unique(vertices, axis=0)  # Get unique vertices
    
    corners = corners[:,[0,2]] - center_2d  # Project to xz plane and center

    ## Iterative closest point
    max_iter = 3
    dist_to_discard = 0.15
    for _ in range(max_iter):
        scale_sum, new_contour, ordered_corners = 0, [], []
        for conpt in contour:
            distance = np.array([np.linalg.norm(conpt - c, ord=2) for c in corners]) # 1d array
            min_index = np.argmin(distance)
            if distance[min_index] > dist_to_discard: continue # no matching mesh corner points, discard
            
            new_contour.append(conpt) # keep it in next iteration
            ordered_corners.append(corners[min_index]) # for if we break
            scale_sum += np.linalg.norm(corners[min_index]) / np.linalg.norm(conpt) # we take its average

        new_contour = np.array(new_contour)
        try:
            transform_scale = scale_sum/new_contour.shape[0]
        except:
            print("scene_id: ", scene_data["scene_id"])
            # save the failed scene in log
            with open("failed_scene_ids.txt", "a") as f:
                f.write(f"{scene_data['scene_id']}\n")
            return None
        if abs(transform_scale-1) < 0.01: break
        contour = new_contour*transform_scale # transform

    ordered_unique_idx = sorted(np.unique(ordered_corners, axis=0, return_index=True)[1])
    ordered_corners = np.array([ordered_corners[i] for i in ordered_unique_idx])

    # Add back the original center before returning and center based on floor plan centroid
    if key_prefix in ["window", "door"]:
        floor_plan_centroid_2d = np.array([scene_data["floor_plan_centroid"][0], scene_data["floor_plan_centroid"][2]])
        ordered_corners = ordered_corners + center_2d - floor_plan_centroid_2d
    # ordered_corners = ordered_corners + center_2d

    if ordered_corners.shape[0] > corners.shape[0]:
        print("Received {} floor plan vertices but found {} ordered corners."\
              .format(corners.shape[0], ordered_corners.shape[0]))
    
    return ordered_corners


def fp_line_normal(fpoc, arch_type="floor", scene_data=None):
    """
    fpoc: [numpt, 2] np array, ordered_corners
    arch_type: "floor", "door", or "window"
    scene_data: needed for door/window to determine wall direction
    """
    fp_line_n = np.zeros((fpoc.shape[0], 2))

    if arch_type in ["door", "window"]:
        # Get floor corners and element center
        floor_corners = scene_data["floor_plan_ordered_corners"]
        element_center = np.mean(fpoc, axis=0)
        
        # Find the longest boundary of the element (should be parallel to wall)
        max_len = 0
        element_dir = None
        for i in range(len(fpoc)):
            start = fpoc[i]
            end = fpoc[(i + 1) % len(fpoc)]
            line_vec = end - start
            line_len = np.linalg.norm(line_vec)
            if line_len > max_len:
                max_len = line_len
                element_dir = line_vec / line_len  # Normalized direction
        
        # Find floor boundary most parallel to element's long side
        min_angle_diff = float('inf')
        wall_normal = None
        
        for i in range(len(floor_corners)):
            start = floor_corners[i]
            end = floor_corners[(i + 1) % len(floor_corners)]
            wall_vec = end - start
            wall_len = np.linalg.norm(wall_vec)
            if wall_len == 0: continue
            
            wall_dir = wall_vec / wall_len
            # Compare directions using dot product
            angle_diff = abs(abs(np.dot(wall_dir, element_dir)) - 1)  # 0 if parallel
            
            if angle_diff < min_angle_diff:
                min_angle_diff = angle_diff
                # Get wall normal (perpendicular to wall)
                wall_normal = np.array([-wall_dir[1], wall_dir[0]])
                # Make sure normal points toward element (will be flipped later)
                to_element = element_center - (start + end) / 2
                if np.dot(wall_normal, to_element) < 0:  # If points away from element
                    wall_normal = -wall_normal
        
        # # Flip normal to point outward from element
        # wall_normal = -wall_normal
        
        # Set all normals to this direction
        for i in range(fpoc.shape[0]):
            fp_line_n[i] = wall_normal

    else:  # Floor plan - keep original inward-pointing normals
        for i in range(fpoc.shape[0]):
            line_vec = fpoc[(i+1)%fpoc.shape[0]] - fpoc[i]
            line_len = np.linalg.norm(line_vec)
            if line_len == 0: 
                print("! fp_line_normal: line_len==0!")
                continue
            fp_line_n[i, 0] = line_vec[1]/line_len    # dy for x axis
            fp_line_n[i, 1] = -line_vec[0]/line_len   # -dx for y axis

    return fp_line_n


def scene_sample_fpbp(fpoc, arch_type="floor", num_sampled_points=256, scene_data=None):
    """ fpoc: [numpt, 2] np array, scene_data's floor_plan_ordered_corners.
        Return sample floor plan boundary pt + normal.
    """
    nfpbp = num_sampled_points

    x = np.append(fpoc[:,0], [fpoc[0,0]])
    y = np.append(fpoc[:,1], [fpoc[0,1]])
    fp_line_n = fp_line_normal(fpoc, arch_type, scene_data)  # Pass arch_type to get correct normal direction

    # sample nfpbp points randomly from the contour outline:
    # Linear length on the line
    dist_bins = np.cumsum( np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ) ) # (nfpoc+1,) cumulative line seg len from bottom left pt
    dist_bins = dist_bins/dist_bins[-1] # (nfpoc+1,), [0, ..., 1] (values normalized to 0 to 1)

    fx, fy = interp1d(dist_bins, x), interp1d(dist_bins, y) # [0, 1] -> [-3/6, 3/6]

    seg_len = float(1)/nfpbp # total perimeter normalized to 1 (distance above)
    seg_starts = np.linspace(0, 1, nfpbp+1)[:-1] # (nfpbp,), starting point of each segment # [0.   0.25 0.5  0.75 1.  ][:-1]
    per_seg_displacement = np.random.uniform(low=0.0, high=seg_len, size=(nfpbp)) # one for each line segment
    sampled_distance = seg_starts + per_seg_displacement # (nfpbp=250, 1)
    sampled_x, sampled_y = fx(sampled_distance), fy(sampled_distance) # (nfpbp=250,), in [-3,3] (convert from 1d sampling to xy coord)

    fpbp = np.concatenate([np.expand_dims(sampled_x, axis=1), np.expand_dims(sampled_y, axis=1)], axis=1) #(nfpbp, 1+1=2)

    bin_idx = np.digitize(sampled_distance, dist_bins) # bins[inds[n]-1] <= x[n] < bins[inds[n]]
    bin_idx -= 1 # (nfpbp=250,) in range [0, nline-1] # example: [ 0  7 10 12 14 17 18 21 22 24] 
    fpbp_normal = fp_line_n[bin_idx, :] # fp_line_n: [nline, 2] -> fpbp_normal: [nfpbp, 2] 

    return np.concatenate([fpbp, fpbp_normal], axis=1) # (nfpbp, 2+2=4)
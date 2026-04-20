'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import torch

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: torch.Tensor) -> List[List[float]]:
    """
    Args:
        img : input image is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    detection_results: List[List[float]] = []

    ##### YOUR IMPLEMENTATION STARTS HERE #####

    return detection_results



def cluster_faces(imgs: Dict[str, torch.Tensor], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    cluster_results: List[List[str]] = [[] for _ in range(K)] # Please make sure your output follows this data format.
        
    ##### YOUR IMPLEMENTATION STARTS HERE #####
    
    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# TODO: Your functions. (if needed)

def _prepare_image_for_face_recognition(img: torch.Tensor) -> torch.Tensor:
    image = img.detach().cpu()

    if image.dim() != 3:
        raise RuntimeError("Expected a 3D image tensor.")

    if image.shape[0] == 3:
        image = image.permute(1, 2, 0)
    elif image.shape[-1] != 3:
        raise RuntimeError("Expected an image with 3 channels.")

    if image.dtype != torch.uint8:
        image = image.float()
        if float(image.max().item()) <= 1.0:
            image = image * 255.0
        image = image.clamp(0.0, 255.0).to(torch.uint8)

    return image.contiguous()


def _tensor_image_to_numpy(img: torch.Tensor):
    return img.contiguous().numpy()


def _flip_last_channel(img: torch.Tensor) -> torch.Tensor:
    return torch.flip(img, dims=(2,))

def _face_location_to_xywh(box, image_h: int, image_w: int) -> List[float]:
    top, right, bottom, left = box

    left = float(max(0, min(int(left), image_w)))
    top = float(max(0, min(int(top), image_h)))
    right = float(max(0, min(int(right), image_w)))
    bottom = float(max(0, min(int(bottom), image_h)))

    width = max(0.0, right - left)
    height = max(0.0, bottom - top)

    return [left, top, width, height]


def _box_area(box) -> float:
    top, right, bottom, left = box
    width = max(0.0, float(right) - float(left))
    height = max(0.0, float(bottom) - float(top))
    return width * height


def _pick_largest_box(boxes: List) -> List:
    if len(boxes) == 0:
        return []

    best_box = boxes[0]
    best_area = _box_area(best_box)
    for box in boxes[1:]:
        area = _box_area(box)
        if area > best_area:
            best_box = box
            best_area = area
    return [best_box]

def _encoding_to_tensor(encoding) -> torch.Tensor:
    tensor = torch.as_tensor(encoding, dtype=torch.float32).reshape(-1)

    if tensor.numel() < 128:
        padded = torch.zeros(128, dtype=torch.float32)
        padded[:tensor.numel()] = tensor
        return padded

    if tensor.numel() > 128:
        tensor = tensor[:128]

    return tensor.contiguous()


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    norms = torch.norm(x, dim=1, keepdim=True).clamp_min(1e-12)
    return x / norms


def _pairwise_squared_distance(points: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    diff = points.unsqueeze(1) - centers.unsqueeze(0)
    return torch.sum(diff * diff, dim=2)


def _initialize_centers(points: torch.Tensor, K: int) -> torch.Tensor:
    num_points = int(points.shape[0])
    dim = int(points.shape[1])
    centers = torch.zeros((K, dim), dtype=points.dtype)

    if num_points == 0:
        return centers

    centers[0] = points[0]
    if K == 1:
        return centers

    min_dist = torch.sum((points - centers[0]) ** 2, dim=1)

    for center_id in range(1, K):
        next_index = int(torch.argmax(min_dist).item())
        centers[center_id] = points[next_index]
        next_dist = torch.sum((points - centers[center_id]) ** 2, dim=1)
        min_dist = torch.minimum(min_dist, next_dist)

    return centers


def _repair_empty_clusters(assignments: torch.Tensor, distances: torch.Tensor, K: int) -> torch.Tensor:
    num_points = int(assignments.shape[0])
    if num_points == 0:
        return assignments

    fixed = assignments.clone()

    for cluster_id in range(K):
        if int((fixed == cluster_id).sum().item()) > 0:
            continue

        donor_cluster = -1
        donor_size = 0
        for current_cluster in range(K):
            current_size = int((fixed == current_cluster).sum().item())
            if current_size > donor_size:
                donor_cluster = current_cluster
                donor_size = current_size

        if donor_cluster < 0 or donor_size <= 1:
            continue

        donor_mask = fixed == donor_cluster
        donor_indices = torch.nonzero(donor_mask, as_tuple=False).reshape(-1)
        donor_dist = distances[donor_indices, donor_cluster]
        move_index = donor_indices[int(torch.argmax(donor_dist).item())]
        fixed[move_index] = cluster_id

    return fixed


def _run_kmeans(points: torch.Tensor, K: int) -> torch.Tensor:
    num_points = int(points.shape[0])
    if num_points == 0:
        return torch.zeros((0,), dtype=torch.long)

    if K == 1:
        return torch.zeros((num_points,), dtype=torch.long)

    centers = _initialize_centers(points, K)
    previous_assignments = None

    for _ in range(50):
        distances = _pairwise_squared_distance(points, centers)
        assignments = torch.argmin(distances, dim=1)
        assignments = _repair_empty_clusters(assignments, distances, K)

        if previous_assignments is not None and torch.equal(assignments, previous_assignments):
            break

        new_centers = centers.clone()
        for cluster_id in range(K):
            mask = assignments == cluster_id
            if bool(mask.any()):
                new_centers[cluster_id] = points[mask].mean(dim=0)

        centers = _normalize_rows(new_centers)
        previous_assignments = assignments.clone()

    final_distances = _pairwise_squared_distance(points, centers)
    final_assignments = torch.argmin(final_distances, dim=1)
    final_assignments = _repair_empty_clusters(final_assignments, final_distances, K)
    return final_assignments.long()

def _find_face_locations(img: torch.Tensor) -> List:
    best_boxes = []
    img_list = [img, _flip_last_channel(img)]

    for now_img in img_list:
        image_np = _tensor_image_to_numpy(now_img)

        for upsample in (0, 1, 2):
            try:
                boxes = face_recognition.face_locations(
                    image_np,
                    number_of_times_to_upsample=upsample,
                    model="hog",
                )
            except Exception:
                boxes = []

            if len(boxes) > len(best_boxes):
                best_boxes = boxes

    return best_boxes
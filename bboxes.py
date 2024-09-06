from lat_lon_distance2 import lat_lon_distance as distance


def point_is_inside_bbox(point, bbox):
    x, y = point
    xmin, ymin, xmax, ymax = bbox
    return (xmin <= x <= xmax) and (ymin <= y <= ymax)

def dmin(bbox_A, bbox_B, metric):
    x_extremeA, y_extremeA, x_extremeB, y_extremeB = find_dmin_extreme_vertices(bbox_A, bbox_B)
    if x_extremeA is None:
        return 0.0
    else:
        return distance(y_extremeA, x_extremeA, y_extremeB, x_extremeB, metric)
    
def find_dmin_extreme_vertices(bbox_A, bbox_B):
    # returns a tuple (x_extremeA, y_extremeA, x_extremeB, y_extremeB) or a None
    xminA, yminA, xmaxA, ymaxA = bbox_A
    xminB, yminB, xmaxB, ymaxB = bbox_B

    cx1 = (xminB<=xminA<=xmaxB) # first possible case for x's ranges intersecting
    cx2 = (xminB<=xmaxA<=xmaxB)  # second possible case for x's ranges intersecting
    cx3 = (xminA<=xminB<=xmaxB<=xmaxA) # last possible case for x's ranges intersecting
    cy1 = (yminB<=yminA<=ymaxB) # first possible case for y's ranges intersecting
    cy2 = (yminB<=ymaxA<=ymaxB) # second possible case for y's ranges intersecting
    cy3 = (yminA<=yminB<=ymaxB<=ymaxA) # last possible case for y's ranges intersecting

    x_extremeA, y_extremeA, x_extremeB, y_extremeB = None, None, None, None
    
    if (cx1 or cx2 or cx3) and (cy1 or cy2 or cy3):
        return (None, None, None, None)
    
    elif (cx1 or cx2 or cx3):
        # that is: if the x's ranges for A and B intersect
        # obs: in this case, it is not possible for the y's ranges to intersect
        # the x_extremeA and x_extremeB will be equal to one another, and equal to any point in the intersection of x's ranges
        if cx1:
            x_extremeA, x_extremeB = xminA, xminA
        elif cx2:
            x_extremeA, x_extremeB = xmaxA, xmaxA
        else: # so cx3
            x_extremeA, x_extremeB = xminB, xminB
        if ymaxA < yminB:
            y_extremeA = ymaxA
            y_extremeB = yminB
        else: # so ymaxB < yminA
            y_extremeA = yminA
            y_extremeB = ymaxB

    elif (cy1 or cy2 or cy3):
        # that is: if the y's ranges for A and B intersect
        # obs: in this case, it is not possible for the x's ranges to intersect
        # the y_extremeA and y_extremeB will be equal to one another, and equal to any point in the intersection of y's ranges
        if cy1:
            y_extremeA, y_extremeB = yminA, yminA
        elif cy2:
            y_extremeA, y_extremeB = ymaxA, ymaxA
        else: # so cy3
            y_extremeA, y_extremeB = yminB, yminB
        if xmaxA < xminB:
            x_extremeA = xmaxA
            x_extremeB = xminB
        else: # so xmaxB < xminA
            x_extremeA = xminA
            x_extremeB = xmaxB
    else: # neither x's ranges or y's ranges intersect
        if xmaxA < xminB:
            x_extremeA = xmaxA
            x_extremeB = xminB
        else: # so xmaxB < xminA
            x_extremeA = xminA
            x_extremeB = xmaxB
        if ymaxA < yminB:
            y_extremeA = ymaxA
            y_extremeB = yminB
        else: # so ymaxB < yminA
            y_extremeA = yminA
            y_extremeB = ymaxB

    return (x_extremeA, y_extremeA, x_extremeB, y_extremeB)

def dmax(bbox_A, bbox_B, metric):
    # first version was based in computations from https://www.cs.mcgill.ca/~fzamal/Project/concepts.htm
    x_extremeA, y_extremeA, x_extremeB, y_extremeB = find_dmax_extreme_vertices(bbox_A, bbox_B)
    return distance(y_extremeA, x_extremeA, y_extremeB, x_extremeB, metric)

def find_dmax_extreme_vertices(bbox_A, bbox_B):
    # return (x_extremeA, y_extremeA, x_extremeB, y_extremeB)
    # Extract coordinates from bounding boxes
    xminA, yminA, xmaxA, ymaxA = bbox_A
    xminB, yminB, xmaxB, ymaxB = bbox_B

    # Calculate corners of the bounding boxes
    cornersA = [(xminA, yminA), (xminA, ymaxA), (xmaxA, yminA), (xmaxA, ymaxA)]
    cornersB = [(xminB, yminB), (xminB, ymaxB), (xmaxB, yminB), (xmaxB, ymaxB)]

    # Find the pair of coordinates with the maximum distance
    max_distance = 0
    max_coordinates = ()

    for cornerA in cornersA:
        for cornerB in cornersB:
            distance = (cornerA[0] - cornerB[0])**2 + (cornerA[1] - cornerB[1])**2

            if distance > max_distance:
                max_distance = distance
                max_coordinates = (*cornerA, *cornerB)

    return max_coordinates

def bboxes_intersect(bbox_A, bbox_B):
    xminA, yminA, xmaxA, ymaxA = bbox_A
    xminB, yminB, xmaxB, ymaxB = bbox_B
    return (xminB<=xminA<=xmaxB or xminB<=xmaxA<=xmaxB or xminA<=xminB<=xmaxA) and \
        (yminB<=yminA<=ymaxB or yminB<=ymaxA<=ymaxB or (yminA<=yminB<=ymaxA))

def intervals_intersect(x1, y1, x2, y2):
    return (x2 <= x1 <= y2) or (x2 <= y1 <= y2) or (x1 <= x2 <= y1)

# def bboxes_intersect(bboxA, bboxB):
#     xa1, ya1, xa2, ya2 = bboxA
#     xb1, yb1, xb2, yb2 = bboxB
#     return intervals_intersect(xa1, xa2, xb1, xb2) and intervals_intersect(ya1, ya2, yb1, yb2)

def is_sub_bbox(bbox_A, bbox_B):
    xminA, yminA, xmaxA, ymaxA = bbox_A
    xminB, yminB, xmaxB, ymaxB = bbox_B
    return (xminB <= xminA and xmaxA <= xmaxB) and (yminB <= yminA and ymaxA <= ymaxB)

def get_children_bboxes(bbox):
    xmin, ymin, xmax, ymax = bbox
    xavg, yavg = (xmin+xmax)/2, (ymin+ymax)/2
    children_bboxes = [
        (xmin, ymin, xavg, yavg),
        (xmin, yavg, xavg, ymax),
        (xavg, ymin, xmax, yavg),
        (xavg, yavg, xmax, ymax)
    ]
    return children_bboxes

def bbox_from_hierarchical_id(hierarchical_id, full_bbox):
    """
    Calculates the bbox of a sub-quadrant given the binary code and the bbox of the parent quadrant.

    Parameters:
    code (str): Binary code of the sub-quadrant.
    bbox (tuple): Bounding box of the parent quadrant (x1, y1, x2, y2).

    Returns:
    tuple: Bounding box of the sub-quadrant (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = full_bbox
    cx, cy = (x1+x2)/2, (y1+y2)/2
    for i in range(0, len(hierarchical_id)):
        c = hierarchical_id[i]
        if c == '0':
            x2 = cx
            y2 = cy
        elif c == '1':
            x2 = cx
            y1 = cy
        elif c == '2':
            x1 = cx
            y2 = cy
        elif c== '3':
            x1 = cx
            y1 = cy
        cx, cy = (x1+x2)/2, (y1+y2)/2
    return (x1, y1, x2, y2)
from pyqtree import Index
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import pyqtree
from collections import defaultdict
from random import random
import numpy as np
from geoobject import GeoObj
from lat_lon_distance2 import lat_lon_distance, get_bbox_by_dist_radius

class ILQuadTree:
    def __init__(self, total_bbox, max_items = 128, max_depth = 3, metric='geodesic'):
        self.quadtrees = dict()
        self.total_bbox = total_bbox
        self.max_items = max_items
        self.max_depth = max_depth
        self.sizes = defaultdict(int)
        self.cached_searches = {}
        self.cached_existence_searches = {}
        self.metric = metric
    
    def insert(self, item, keywords):
        for keyword in keywords:
            if keyword not in self.quadtrees:
                self.quadtrees[keyword] = Index(bbox=self.total_bbox, max_depth = self.max_depth, max_items=self.max_items) #max_items = self.max_items,
            self.sizes[keyword] += 1
            self.quadtrees[keyword].insert(item, item.bbox())

    def clean_caches(self):
        self.cached_searches = {}
        self.cached_existence_searches = {}

    def add_cached_search(self, keyword, center, radius, result):
        self.cached_searches[(keyword, center, radius)] = result

    def add_cached_existence_search(self, keyword, center, radius, result):
        self.cached_existence_searches[(keyword, center, radius)] = result

    def insert_elements_from_list(self, obj_list):
        for obj in obj_list:
            self.insert(obj, obj.keywords())

    def insert_elements_from_geopandas(self, gdf):
        for i in range(gdf.shape[0]):
            obj = GeoObj(gdf.iloc[i])
            self.insert(obj, obj.keywords())
            #self.insert(gdf.iloc[i], set(gdf.iloc[i][['amenity','shop','tourism','landuse','leisure','building','office','government']].values))
    
    def search_bbox(self, keywords, bbox):
        result = []
        for keyword in keywords:
            if keyword in self.quadtrees:
                #for item in self.quadtrees[keyword].intersect(bbox):
                result.extend(self.quadtrees[keyword].intersect(bbox))
                    #result.add(item)
        return result
    
    def search_circle(self, keywords, center, radius):
        result = []
        # ERRO: a bbox a ser analisada não é essa: não se subtrai metros de lat, long
        bbox = get_bbox_by_dist_radius(center, radius, self.metric)
        #bbox = [center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius]
        for keyword in keywords:
            if keyword in self.quadtrees:
                for item in self.quadtrees[keyword].intersect(bbox):
                    #xmin,ymin,xmax,ymax = item.bbox()
                    #item_center = (ymin+ymax)/2, (xmin+xmax)/2
                    item_center_lon, item_center_lat = item.centroid()
                    if lat_lon_distance(item_center_lat, item_center_lon, center[1], center[0], self.metric) <= radius:
                        result.append(item)
        return result

    def search_circle_existence(self, keyword, center, radius):
        result = []
        # ERRO: a bbox a ser analisada não é essa: não se subtrai metros de lat, long
        bbox = get_bbox_by_dist_radius(center, radius, self.metric)
        #bbox = [center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius]
        if keyword in self.quadtrees:
            for item in self.quadtrees[keyword].intersect(bbox):
                #xmin,ymin,xmax,ymax = item.bbox()
                #item_center = (ymin+ymax)/2, (xmin+xmax)/2
                item_center_lon, item_center_lat = item.centroid()
                if lat_lon_distance(item_center_lat, item_center_lon, center[1], center[0], self.metric) <= radius:
                    return True
        return False
    
    def get_obj_by_keyword_and_osmid(self, keyword, osmid):
        if keyword in self.quadtrees:
            objs = get_objects(self.quadtrees[keyword])
            return list(filter(lambda s: s[0].item['osm_id'] == osmid, objs))
        else:
            return []
    
    def plot_subdivisions(self, quadtree, bbox, ax):
        xmin, ymin, xmax, ymax = bbox
        ax.plot([xmin,xmax,xmax,xmin,xmin], [ymin,ymin,ymax,ymax,ymin], c='k')
        if self.is_subdivided(quadtree):
            self.plot_subdivisions(quadtree.children[0], self.bbox_from_code('00', bbox), ax)
            self.plot_subdivisions(quadtree.children[1], self.bbox_from_code('10', bbox), ax)
            self.plot_subdivisions(quadtree.children[2], self.bbox_from_code('01', bbox), ax)
            self.plot_subdivisions(quadtree.children[3], self.bbox_from_code('11', bbox), ax)
    
    def plot(self, keyword, point_size = 50, point_color='black', point_shape = 'point', subtrees_to_highlight = []):
        from matplotlib import pyplot as plt
        fig = plt.figure(figsize = (10,7))
        ax = plt.subplot()
        if keyword in self.quadtrees:
            qt = self.quadtrees[keyword]
        else:
            qt = None
        self.plot_quadtree(qt, self.total_bbox, ax, point_size=point_size, point_color=point_color, point_shape=point_shape, subtrees_to_highlight=subtrees_to_highlight)
        xmin,ymin,xmax,ymax = self.total_bbox
        plt.xlim(xmin - 0.001, xmax + 0.001)
        plt.ylim(ymin - 0.001, ymax + 0.001)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.show()

    def plot_geometries(self, quadtree, bbox, ax, point_size, point_color='black', point_shape='point'):
        markers = {
            'point': 'o',
            'triangle': '^',
            'square': 's',
            'star': '*'
        }
        for node in quadtree.nodes:
            x1, y1, x2, y2 = node.rect
            rect = Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='blue')
            ax.add_patch(rect)
            if x1==x2 and y1==y2:
                ax.scatter([x1,x2], [y1,y2], s=point_size, c=point_color, marker = markers[point_shape])
        if self.is_subdivided(quadtree):
            self.plot_geometries(quadtree.children[0], self.bbox_from_code('00', bbox), ax, point_size,point_color,point_shape)
            self.plot_geometries(quadtree.children[1], self.bbox_from_code('10', bbox), ax, point_size,point_color,point_shape)
            self.plot_geometries(quadtree.children[2], self.bbox_from_code('01', bbox), ax, point_size,point_color,point_shape)
            self.plot_geometries(quadtree.children[3], self.bbox_from_code('11', bbox), ax, point_size,point_color,point_shape)
        
    def plot_quadtree(self, quadtree, bbox, ax, point_size = 50, point_color = 'b', point_shape='point', subtrees_to_highlight = []):
        if quadtree is None:
            print('Quadtree is None')
            return
        bboxes_to_highlight = [bbox_from_code(s, bbox) for s in subtrees_to_highlight]
        self.plot_subdivisions(quadtree, bbox, ax)
        self.plot_geometries(quadtree, bbox, ax, point_size, point_color, point_shape)
        # highlight subtrees from codes:
        for bth in bboxes_to_highlight:
            x1, y1, x2, y2 = bth
            ax.plot([x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1], c='r')
            if x1==x2 and y1==y2:
                ax.scatter([x1,x2], [y1,y2], c='r')
        # for node in quadtree.nodes:
        #     x1, y1, x2, y2 = node.rect
        #     rect = Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='blue')
        #     ax.add_patch(rect)
        #     if x1==x2 and y1==y2:
        #         ax.scatter([x1,x2], [y1,y2], c='b')
        # if self.is_subdivided(quadtree):
        #     self.plot_quadtree(quadtree.children[0], self.bbox_from_code('00', bbox), ax, subtrees_to_highlight)
        #     self.plot_quadtree(quadtree.children[1], self.bbox_from_code('10', bbox), ax, subtrees_to_highlight)
        #     self.plot_quadtree(quadtree.children[2], self.bbox_from_code('01', bbox), ax, subtrees_to_highlight)
        #     self.plot_quadtree(quadtree.children[3], self.bbox_from_code('11', bbox), ax, subtrees_to_highlight)
    

    

    def bbox_to_quadrant(self, bbox):
        """
        Converte um bounding box em coordenadas de quadrante na IL-Quadtree.
        """
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        radius = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 2
        return (cx, cy, radius)
    
    def bbox_from_code(self, code, bbox):
        """
        Calculates the bbox of a sub-quadrant given the binary code and the bbox of the parent quadrant.

        Parameters:
        code (str): Binary code of the sub-quadrant.
        bbox (tuple): Bounding box of the parent quadrant (x1, y1, x2, y2).

        Returns:
        tuple: Bounding box of the sub-quadrant (x1, y1, x2, y2).
        """
        x1, y1, x2, y2 = bbox
        cx, cy = (x1+x2)/2, (y1+y2)/2
        width, height = abs(x2-x1), abs(y2-y1)
        for i in range(0, len(code), 2):
            c = code[i:i+2]
            if c == '00':
                x2 = cx
                y2 = cy
            elif c == '01':
                x1 = cx
                y2 = cy
            elif c == '10':
                x2 = cx
                y1 = cy
            elif c== '11':
                x1 = cx
                y1 = cy
            cx, cy = (x1+x2)/2, (y1+y2)/2
            width, height = abs(x2-x1), abs(y2-y1)
        return (x1, y1, x2, y2)
    
    def sub_tree_from_code(self, quadtree, code):
        try:
            sub_tree = quadtree
            for i in range(0, len(code)//2+1, 2):
                c = code[i:i+2]
                if c == '00':
                    sub_tree = sub_tree.children[0]
                elif c == '01':
                    sub_tree = sub_tree.children[2]
                elif c == '10':
                    sub_tree = sub_tree.children[1]
                elif c== '11':
                    sub_tree = sub_tree.children[3]
        except IndexError:
            print('Invalid code: path doesn\'t exist!')
            return
            
        return sub_tree
    
    def is_subdivided(self, quadtree):
        return quadtree.children != []
    
    def get_leaves(self, quadtree):
        leaves = quadtree.nodes.copy()
        if self.is_subdivided(quadtree):
            for child in quadtree.children:
                leaves.extend(self.get_leaves(child))
        return leaves
            
        
    def get_depth(self):
        depths = [get_depth(quadtree) for quadtree in self.quadtrees.values()]
        return max(depths)
        
    def get_objects(self):
        objects = []
        for qtree in self.quadtrees.values():
            objects.extend(get_objects(qtree))
        return list(set(objects))
    
    def get_object_by_id(self, id):
        objects = self.get_objects()
        obj = GeoObj.get_object_by_id(objects, id)
        return obj
        
    
    def display_objects(self):
        fig = plt.figure(figsize = (10,7))
        ax = plt.subplot()
        xmin,ymin,xmax,ymax = self.total_bbox
        range_x = xmax-xmin
        range_y = ymax-ymin
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        objects = self.get_objects()
        keywords_frequencies = list(self.sizes.items())
        keywords_frequencies.sort(key = lambda e: e[1])
        keywords_frequencies.reverse()
        keywords_frequencies = keywords_frequencies[:30]
        keywords = [k for k,_ in keywords_frequencies]
        keywords.append('other')
        ploted_keywords = []
        colors = {keyword: [(random(),random(),random())] for keyword in keywords}
        for i,obj in enumerate(objects):
            for keyword in obj.keywords:
                if keyword not in keywords:
                    keyword = 'other'
                label = None
                if keyword not in ploted_keywords:
                    label = keyword
                    ploted_keywords.append(keyword)
                ax.scatter(*jitter(obj.x, obj.y, range_x, range_y), c = colors[keyword], label = label)
        plt.legend()
        plt.show()
            

def is_subdivided(quadtree):
    return len(quadtree.children)>0

def get_depth(quadtree):
    depth = 0
    if is_subdivided(quadtree):
        depth = 1 + max([get_depth(c) for c in quadtree.children])
    return depth
        

def get_MBR(quadtree):
    x,y = quadtree.center
    xmin, xmax = x-quadtree.width/2 , x+quadtree.width/2
    ymin, ymax = y-quadtree.height/2 , y+quadtree.height/2
    return [xmin, ymin, xmax, ymax]

def construct_nodes_bboxes_from_quadtree(quadtree, quadtree_code_id = '', nodes_info = None):
    if nodes_info is None:
        nodes_info = []
    nodes_info.append(
        {'subquadtree': quadtree, 
         'subquadtree_code': quadtree_code_id, 
         'bbox': get_MBR(quadtree), 
         'total_children_nodes': len(quadtree.children), 
         'total_inner_objects': len(quadtree.nodes)}
    )
    child_nodes = quadtree.children.copy()
    if len(child_nodes)>0:
        for i, node in enumerate(child_nodes.copy()):
            construct_nodes_bboxes_from_quadtree(node, quadtree_code_id = quadtree_code_id+str(i), nodes_info = nodes_info)
    return nodes_info

def point_is_inside_bbox(point, bbox):
    x, y = point
    xmin, ymin, xmax, ymax = bbox
    return (xmin <= x <= xmax) and (ymin <= y <= ymax)

def dmin(bbox_A, bbox_B, metric):
    # from shapely import Polygon
    # xminA, yminA = bbox_A[0], bbox_A[1]
    # xmaxA, ymaxA = bbox_A[2], bbox_A[3]
    # xminB, yminB = bbox_B[0], bbox_B[1]
    # xmaxB, ymaxB = bbox_B[2], bbox_B[3]
    # pa = Polygon([(xminA,yminA),(xmaxA,yminA), (xmaxA, ymaxA), (xminA,ymaxA)])
    # pb = Polygon([(xminB,yminB),(xmaxB,yminB), (xmaxB, ymaxB), (xminB,ymaxB)])
    # return pa.distance(pb)
    x_extremeA, y_extremeA, x_extremeB, y_extremeB = find_dmin_extreme_vertices(bbox_A, bbox_B)
    #print('in dmin')
    #print(x_extremeA, y_extremeA, x_extremeB, y_extremeB)
    # if x_extremeA is None:
    #     print('x_extremeA is None')
    #     return 0.0
    # if y_extremeA is None:
    #     print('y_extremeA is None')
    # if x_extremeB is None:
    #     print('x_extremeB is None')
    # if y_extremeB is None:
    #     print('y_extremeB is None')
    # print('end in dmin')
    if x_extremeA is None:
        return 0.0
    else:
        #print('here:', y_extremeA, x_extremeA, y_extremeB, x_extremeB)
        return lat_lon_distance(y_extremeA, x_extremeA, y_extremeB, x_extremeB, metric)
    
def find_dmin_extreme_vertices(bbox_A, bbox_B):
    # return (x_extremeA, y_extremeA, x_extremeB, y_extremeB) or None
    xminA, yminA = bbox_A[0], bbox_A[1]
    xmaxA, ymaxA = bbox_A[2], bbox_A[3]
    xminB, yminB = bbox_B[0], bbox_B[1]
    xmaxB, ymaxB = bbox_B[2], bbox_B[3]

    cx1 = (xminB<=xminA<=xmaxB) # first possible case for x's ranges intersecting
    cx2 = (xminB<=xmaxA<=xmaxB)  # second possible case for x's ranges intersecting
    cx3 = (xminA<=xminB<=xmaxB<=xmaxA) # last possible case for x's ranges intersecting
    cy1 = (yminB<=yminA<=ymaxB) # first possible case for y's ranges intersecting
    cy2 = (yminB<=ymaxA<=ymaxB) # second possible case for y's ranges intersecting
    cy3 = (yminA<=yminB<=ymaxB<=ymaxA) # last possible case for y's ranges intersecting

    x_extremeA, y_extremeA, x_extremeB, y_extremeB = None, None, None, None
    
    if (cx1 or cx2 or cx3) and (cy1 or cy2 or cy3):
        #print('case11')
        return (None, None, None, None)
    
    elif (cx1 or cx2 or cx3):
        #print('case22')
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
        #print('case33')
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
        #print('case44')
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

    # if x_extremeA is None:
    #     print('x_extremeA is None')
    # if y_extremeA is None:
    #     print('y_extremeA is None')
    # if x_extremeB is None:
    #     print('x_extremeB is None')
    # if y_extremeB is None:
    #     print('y_extremeB is None')
    return (x_extremeA, y_extremeA, x_extremeB, y_extremeB)

def dmax(bbox_A, bbox_B, metric):
    #baseado em calculos de https://www.cs.mcgill.ca/~fzamal/Project/concepts.htm
    # xminA, yminA = bbox_A[0], bbox_A[1]
    # xmaxA, ymaxA = bbox_A[2], bbox_A[3]
    # xminB, yminB = bbox_B[0], bbox_B[1]
    # xmaxB, ymaxB = bbox_B[2], bbox_B[3]
    # delta_x = max(abs(xminA - xmaxB) , abs(xmaxA - xminB))
    # delta_y = max(abs(yminA - ymaxB) , abs(ymaxA - yminB))
    # return (delta_x**2 + delta_y**2)**(1/2)
    x_extremeA, y_extremeA, x_extremeB, y_extremeB = find_dmax_extreme_vertices(bbox_A, bbox_B)
    return lat_lon_distance(y_extremeA, x_extremeA, y_extremeB, x_extremeB, metric)

def bboxes_intersect(bbox_A, bbox_B):
    xminA, yminA = bbox_A[0], bbox_A[1]
    xmaxA, ymaxA = bbox_A[2], bbox_A[3]
    xminB, yminB = bbox_B[0], bbox_B[1]
    xmaxB, ymaxB = bbox_B[2], bbox_B[3]
    return (xminB<=xminA<=xmaxB or xminB<=xmaxA<=xmaxB or xminA<=xminB<=xmaxA) and \
            (yminB<=yminA<=ymaxB or yminB<=ymaxA<=ymaxB or (yminA<=yminB<=ymaxA))

    
def get_objects(quadtree, current_node_id = ''):
    leaves = list(zip([e.item for e in quadtree.nodes], len(quadtree.nodes)*[current_node_id]))
    if len(quadtree.children)>0:
        for i, child in enumerate(quadtree.children):
            leaves.extend(get_objects(child, current_node_id + str(i)))
    return leaves

def get_leaves(quadtree):
    leaves = quadtree.nodes.copy()
    if len(quadtree.children)>0:
        for child in quadtree.children:
            leaves.extend(get_leaves(child))
    return leaves
                
# def get_objects(quadtree):
#     objects = list(zip(*get_leaves(quadtree)))[0]
#     for i,leaf in enumerate(objects):
#         objects[i] = leaf.item
#     return objects

def get_size(quadtree):
    return len(get_objects(quadtree))

def get_nodes(quadtree):
    nodes = quadtree.children.copy()
    if len(nodes)>0:
        for node in nodes.copy():
            nodes.extend(get_nodes(node))
    return nodes

def get_nodes_at_level(quadtree, level: int):
    depth = get_depth(quadtree)
    if level > depth:
        level = depth
    if level == 0:
        return [quadtree]

    nodes = defaultdict(list)
    for l in range(1, level+1):
        # calcular os nodes do level l
        nodes_previous_level = get_nodes_at_level(quadtree, l-1)
        for node in nodes_previous_level:
            nodes[l].extend(node.children)
    return nodes[level]


# def search_circle_quadtree(quadtree, center, radius, metric):
#     result = []
#     bbox = [center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius]
#     search_area = quadtree.intersect(bbox)
#     for item in search_area:
#         xmin,ymin,xmax,ymax = item.bbox()
#         xc, yc = (xmin+xmax)/2, (ymin+ymax)/2
#         if lat_lon_distance(yc, xc, center[1], center[0], metric) <= radius:
#             result.append(item)
#     return result

def jitter(x, y, range_x, range_y):
    arr = np.array([x,y])
    stdev = .0075 * np.array([range_x, range_y])
    return (arr + np.random.randn(*arr.shape) * stdev).tolist()

def bbox_from_code(code, bbox):
    """
    Calculates the bbox of a sub-quadrant given the binary code and the bbox of the parent quadrant.

    Parameters:
    code (str): Binary code of the sub-quadrant.
    bbox (tuple): Bounding box of the parent quadrant (x1, y1, x2, y2).

    Returns:
    tuple: Bounding box of the sub-quadrant (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1+x2)/2, (y1+y2)/2
    width, height = abs(x2-x1), abs(y2-y1)
    for i in range(0, len(code), 2):
        c = code[i:i+2]
        if c == '00':
            x2 = cx
            y2 = cy
        elif c == '01':
            x1 = cx
            y2 = cy
        elif c == '10':
            x2 = cx
            y1 = cy
        elif c== '11':
            x1 = cx
            y1 = cy
        cx, cy = (x1+x2)/2, (y1+y2)/2
        width, height = abs(x2-x1), abs(y2-y1)
    return (x1, y1, x2, y2)
    
def sub_tree_from_code(quadtree, code):
    try:
        sub_tree = quadtree
        for i in range(0, len(code)//2+1, 2):
            c = code[i:i+2]
            if c == '00':
                sub_tree = sub_tree.children[0]
            elif c == '01':
                sub_tree = sub_tree.children[2]
            elif c == '10':
                sub_tree = sub_tree.children[1]
            elif c== '11':
                sub_tree = sub_tree.children[3]
    except IndexError:
        print('Invalid code: path doesn\'t exist!')
        return

    return sub_tree

def plot_subdivisions(quadtree, bbox, ax):
    xmin, ymin, xmax, ymax = bbox
    ax.plot([xmin,xmax,xmax,xmin,xmin], [ymin,ymin,ymax,ymax,ymin], c='k')
    if is_subdivided(quadtree):
        plot_subdivisions(quadtree.children[0], bbox_from_code('00', bbox), ax)
        plot_subdivisions(quadtree.children[1], bbox_from_code('10', bbox), ax)
        plot_subdivisions(quadtree.children[2], bbox_from_code('01', bbox), ax)
        plot_subdivisions(quadtree.children[3], bbox_from_code('11', bbox), ax)

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
    # xminA, yminA = bbox_A[0], bbox_A[1]
    # xmaxA, ymaxA = bbox_A[2], bbox_A[3]
    # xminB, yminB = bbox_B[0], bbox_B[1]
    # xmaxB, ymaxB = bbox_B[2], bbox_B[3]
    # if xminA < xminB:
    #     x_extremeA = xminA
    #     x_extremeB = xmaxB
    # else:
    #     x_extremeA = xmaxA
    #     x_extremeB = xminB
    # if yminA < yminB:
    #     y_extremeA = yminA
    #     y_extremeB = ymaxB
    # else:
    #     y_extremeA = ymaxA
    #     y_extremeB = yminB
    # return (x_extremeA, y_extremeA, x_extremeB, y_extremeB)



def plot_node_pairs(quadtree1, quadtree2, code_pairs = []):
    fig, ax = plt.subplots(max(len(code_pairs),1),2)
    fig.set_figwidth(7)
    fig.set_figheight(3*max(1,len(code_pairs)))
    if code_pairs == []:
        ax1 = ax[0]; ax1.set_xlim(-0.01,1.01); ax1.set_ylim(-0.01,1.01); ax1.set_xticks([]); ax1.set_yticks([])
        ax2 = ax[1]; ax2.set_xlim(-0.01,1.01); ax2.set_ylim(-0.01,1.01); ax2.set_xticks([]); ax2.set_yticks([])
        plot_subdivisions(quadtree1, [0,0,1,1], ax1)
        plot_subdivisions(quadtree2, [0,0,1,1], ax2)
    else:
        for i, code_pair in enumerate(code_pairs):
            ax1 = ax[i][0]; ax1.set_xlim(-0.01,1.01); ax1.set_ylim(-0.01,1.01); ax1.set_xticks([]); ax1.set_yticks([])
            ax2 = ax[i][1]; ax2.set_xlim(-0.01,1.01); ax2.set_ylim(-0.01,1.01); ax2.set_xticks([]); ax2.set_yticks([])
            plot_subdivisions(quadtree1, [0,0,1,1], ax1)
            plot_subdivisions(quadtree2, [0,0,1,1], ax2)
            bbox1 = bbox_from_code(code_pair[0], [0,0,1,1])
            bbox2 = bbox_from_code(code_pair[1], [0,0,1,1])

            x1, y1, x2, y2 = bbox1
            ax1.plot([x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1], c='r')
            if x1==x2 and y1==y2:
                ax1.scatter([x1,x2], [y1,y2], c='r')

            x1, y1, x2, y2 = bbox2
            ax2.plot([x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1], c='r')
            if x1==x2 and y1==y2:
                ax2.scatter([x1,x2], [y1,y2], c='r')
    plt.show()
    
#get_nodes_at_level(il_quadtree.quadtrees['keyword2'], 2) 
# item1 = {'id':1, 'bbox':[0.2,0.2,0.2,0.2]}
# item2 = {'id':2, 'bbox':[0.4,0.5,0.4,0.5]}
# item3 = {'id':'teste', 'bbox':[0.8,0.8,0.8,0.8]}
# il_quadtree.insert(item1, ['keyword1', 'keyword2'])
# il_quadtree.insert(item2, ['keyword2', 'keyword3'])
# il_quadtree.insert(item3, ['keyword3', 'keyword4'])
# result = il_quadtree.search_bbox(['keyword2'], [0.2, 0.2, 1, 1])
# il_quadtree.quadtrees['keyword2'].children[0].nodes[1].item
# il_quadtree.plot(keywords = ['keyword2'])
# il_quadtree.get_depth(il_quadtree.quadtrees['keyword2'])

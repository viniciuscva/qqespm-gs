import matplotlib.pyplot as plt
from . import remote_quadtree
from random import random
import numpy as np
from lat_lon_distance2 import lat_lon_distance, get_bbox_by_dist_radius
from ilquadtree import ILQuadTree
from functools import partial
import psutil
import gc
import os

def is_within_radius(item, center, radius, metric):
    item_center_lon, item_center_lat = item.centroid()
    return lat_lon_distance(item_center_lat, item_center_lon, center[1], center[0], metric) <= radius


def get_percent_memory_used():
    total_mem = psutil.virtual_memory().total# /(2**20)
    used_mem = psutil.virtual_memory().used# /(2**20)
    return used_mem/total_mem


class RemoteILQuadtree:
    def __init__(self, ram_ilq: ILQuadTree, remote_ilq_dir, metric = 'geodesic'):
        self.quadtrees = dict()
        self.total_bbox = ram_ilq.total_bbox
        self.remote_ilq_dir = remote_ilq_dir
        self.max_items = ram_ilq.max_items
        self.max_depth = ram_ilq.max_depth
        self.sizes = ram_ilq.sizes
        self.cached_searches = {}
        self.cached_existence_searches = {}
        self.metric = metric

        if not os.path.isdir(remote_ilq_dir):
            os.mkdir(remote_ilq_dir)


        for file in os.listdir(remote_ilq_dir):
            os.remove(f'{remote_ilq_dir}/{file}')
 
        for keyword in ram_ilq.quadtrees:
            ram_quadtree = ram_ilq.quadtrees[keyword]
            objects_remote_dir = f'{remote_ilq_dir}/qtree_{keyword}_'
            self.quadtrees[keyword] = remote_quadtree.RemoteQuadtree(ram_quadtree, objects_remote_dir)

    def get_all_osm_ids_from_ilq_costly(self):
        objs = self.get_objects_costly()
        osm_ids = [o.item['osm_id'] for o in objs]
        return osm_ids

    def clean_caches(self):
        self.cached_searches = {}
        self.cached_existence_searches = {}

    def add_cached_search(self, keyword, center, radius, result):
        self.cached_searches[(keyword, center, radius)] = result

    def add_cached_existence_search(self, keyword, center, radius, result):
        self.cached_existence_searches[(keyword, center, radius)] = result

    def clean_memory(self):
        for keyword in self.quadtrees:
            self.quadtrees[keyword].clean_memory()
        gc.collect()

    def balance_memory_allocation(self, acceptance_threshold = 0.90):
        percent_memory_alloc = get_percent_memory_used()
        if percent_memory_alloc > acceptance_threshold:
            self.clean_memory()
    
    def search_bbox(self, keywords, bbox):
        result = []
        for keyword in keywords:
            if keyword in self.quadtrees:
                result.extend(self.quadtrees[keyword].intersect(bbox))
        return result
    
    def search_circle(self, keyword, center, radius):
        result = []
        bbox = get_bbox_by_dist_radius(center, radius, self.metric)
        if keyword in self.quadtrees:
            candidate_objs = np.array(self.quadtrees[keyword].intersect(bbox))
            if len(candidate_objs) == 0:
                return []
            is_within_radius_partial = partial(is_within_radius, center = center, radius = radius, metric=self.metric)
            is_within_radius_vec = np.vectorize(is_within_radius_partial)
            return candidate_objs[np.where(is_within_radius_vec(candidate_objs))]
        return result

    def search_circle_existence(self, keyword, center, radius):
        bbox = get_bbox_by_dist_radius(center, radius, self.metric)
        if keyword in self.quadtrees:
            candidate_objs = self.quadtrees[keyword].intersect(bbox)
                
            tests = filter(lambda item: lat_lon_distance(*reversed(item.centroid()), center[1], center[0], self.metric) <= radius, candidate_objs)
            for test in tests:
                if test:
                    return True
   
        return False
    
    def get_obj_by_keyword_and_osmid_costly(self, keyword, osmid):
        if keyword in self.quadtrees:
            objs = self.quadtrees[keyword].get_objects_costly()
            return list(filter(lambda s: s.item['osm_id'] == osmid, objs))
        else:
            return []
    
    
    def plot(self, keyword, include_objects_costly = False, hierarchical_ids_to_highlight = None):
        if hierarchical_ids_to_highlight is None:
            hierarchical_ids_to_highlight = []

        from matplotlib import pyplot as plt
        _ = plt.figure(figsize = (10,7))
        ax = plt.subplot()

        if keyword not in self.quadtrees:
            return None
        qtree = self.quadtrees[keyword]
        qtree.plot(ax, include_objects_costly, hierarchical_ids_to_highlight)

        xmin,ymin,xmax,ymax = self.total_bbox
        plt.xlim(xmin - 0.001, xmax + 0.001)
        plt.ylim(ymin - 0.001, ymax + 0.001)
        plt.show()
            
    def get_depth(self):
        depths = [quadtree.get_depth() for quadtree in self.quadtrees.values()]
        return max(depths)
        
    def get_objects_costly(self):
        objects = []
        for qtree in self.quadtrees.values():
            objects.extend(qtree.get_objects_costly())
        return list(set(objects))
    
    def get_object_by_id_costly(self, id_):
        objects = self.get_objects_costly()
        obj = next(filter(lambda obj: obj.item['osm_id'] == id_, objects))
        return obj
    
    def display_objects_costly(self):
        _ = plt.figure(figsize = (10,7))
        ax = plt.subplot()
        xmin,ymin,xmax,ymax = self.total_bbox
        delta_x = xmax-xmin
        delta_y = ymax-ymin
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        objects = self.get_objects_costly()
        keywords_frequencies = list(self.sizes.items())
        keywords_frequencies.sort(key = lambda e: e[1], reverse=True)
        keywords_frequencies = keywords_frequencies[:30]
        keywords = [k for k,_ in keywords_frequencies]
        keywords.append('other')
        ploted_keywords = []
        colors = {keyword: [(random(),random(),random())] for keyword in keywords}
        for obj in objects:
            for keyword in obj.keywords():
                if keyword not in keywords:
                    keyword = 'other'
                label = None
                if keyword not in ploted_keywords:
                    label = keyword
                    ploted_keywords.append(keyword)
                ax.scatter(*_jitter(*obj.centroid(), delta_x, delta_y), c = colors[keyword], label = label)
        plt.legend()
        plt.show()


def _jitter(x, y, delta_x, delta_y, jitter_size = 0.0075):
    arr = np.array([x,y])
    stdev = jitter_size * np.array([delta_x, delta_y])
    return (arr + np.random.randn(*arr.shape) * stdev).tolist()



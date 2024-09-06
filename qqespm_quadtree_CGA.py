from ilquadtree import ILQuadTree
from geoobject import GeoObj
import json
import multiprocessing
from multiprocessing.pool import ThreadPool
#from multiprocessing import Pool
from time import time
import psutil
from functools import partial
from itertools import product as cartesian_product
from lat_lon_distance2 import lat_lon_distance
import itertools
import json
from collections import defaultdict
from remote_ilquadtree import remote_ilquadtree
from bboxes import bboxes_intersect, dmin, dmax
import os
import pickle
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

remote_ilqs = dict()
remote_ilq = None
total_bbox_ilq = None
ilq_object_path = 'remote_ilq_obj.pkl'
current_ilq_dir = None

class hashabledict(dict):
    def __hash__(self):
        return hash(json.dumps(self))
    



def read_df_csv(data_dir = os.path.dirname(os.path.realpath(__file__)) + '/data/pois_paraiba5.csv'):
    import pandas as pd
    import geopandas
    pois = pd.read_csv(data_dir,  low_memory=False)
    pois['geometry'] = geopandas.GeoSeries.from_wkt(pois['geometry'])
    pois['centroid'] = geopandas.GeoSeries.from_wkt(pois['centroid'])
    return pois

def get_df_surrounding_bbox(pois, delta = 0.001):
    lons_lats = np.vstack([np.array(t) for t in pois['centroid'].apply(lambda e: e.coords[0]).values])
    pois['lon'], pois['lat'] = lons_lats[:, 0], lons_lats[:, 1]
    surrounding_bbox = (pois['lon'].min()-delta, pois['lat'].min()-delta, pois['lon'].max()+delta, pois['lat'].max()+delta)
    pois.drop(['lon','lat'], axis = 1, inplace = True)
    return surrounding_bbox

def generate_remote_ilquadtree(data_dir = os.path.dirname(os.path.realpath(__file__)) + '/data/pois_paraiba5.csv', ilq_folder = 'ilq_files', 
                               total_bbox_ilq = None, max_depth = 3, max_items = 128, 
                               metric='geodesic', keyword_columns = ['amenity','shop','tourism','landuse','leisure','building'], insertion_fraction = 1.0):
    ilq = generate_ram_ilquadtree(data_dir = data_dir, total_bbox_ilq = total_bbox_ilq, max_depth = max_depth, max_items = max_items, 
                               metric=metric, keyword_columns = keyword_columns, insertion_fraction = insertion_fraction)
    ilq_remote = remote_ilquadtree.RemoteILQuadtree(ilq, ilq_folder, metric=metric)
    return ilq_remote

def generate_ram_ilquadtree(data_dir = os.path.dirname(os.path.realpath(__file__)) + '/data/pois_paraiba5.csv', total_bbox_ilq = None, max_depth = 3, max_items = 128, 
                               metric='geodesic', keyword_columns = ['amenity','shop','tourism','landuse','leisure','building'], insertion_fraction = 1.0):
    pois = read_df_csv(data_dir=data_dir)
    if total_bbox_ilq is None:
        total_bbox_ilq = get_df_surrounding_bbox(pois)
    objs = GeoObj.get_objects_from_geopandas(pois, keyword_columns = keyword_columns)
    ilq = ILQuadTree(total_bbox = total_bbox_ilq, max_depth = max_depth, max_items=max_items, metric=metric)
    ilq.insert_elements_from_list(objs[0: int(insertion_fraction*len(objs))+1])
    return ilq

def load_remote_ilquadtree(ilq_object_path):
    with open(ilq_object_path, 'rb') as f:
        remote_ilq = pickle.load(f)
    return remote_ilq

def get_keyword_frequency(keyword: str, df, column_names = ['amenity','shop','tourism','landuse','leisure','building']):
    filter_expression = df[column_names[0]]==keyword
    for column in column_names[1:]:
        filter_expression = (filter_expression)|(df[column]==keyword)
    rows_with_keyword = df[filter_expression]
    return rows_with_keyword.shape[0]

def get_keywords_frequencies(keywords, df, column_names = ['amenity','shop','tourism','landuse','leisure','building']):
    keywords_frequencies = {}
    for keyword in keywords:
        keywords_frequencies[keyword] = get_keyword_frequency(keyword, df, column_names)
    return keywords_frequencies

def get_all_osmids_from_ilq_obj_dir(ilq_obj_dir):
    with open(ilq_obj_dir, 'rb') as f:
        ilq_obj = pickle.load(f)
    return ilq_obj.get_all_osm_ids_from_ilq_costly()


class SpatialVertex:
    def __init__(self, id, keyword):
        self.id = id
        self.keyword = keyword
    def __str__(self):
        return '   ' + str(self.id) + ' (' + str(self.keyword) + ')'

    def __hash__(self):
        return hash(self.__str__())

    def __repr__(self):
        return '   ' + str(self.id) + ' (' + str(self.keyword) + ')'

    def __eq__(self, another_vertex):
        return self.id == another_vertex.id and self.keyword == another_vertex.keyword

    def to_json(self):
        return json.dumps(self.to_dict())

    def to_dict(self):
        return {'id': self.id, 'keyword': self.keyword}

    @staticmethod
    def from_json(json_str):
        vertex_dict = json.loads(json_str)
        return SpatialVertex.from_dict(vertex_dict)

    @staticmethod
    def from_dict(vertex_dict):
        id = vertex_dict['id']
        keyword = vertex_dict['keyword']
        return SpatialVertex(id, keyword)

    @staticmethod
    def from_id(id, vertices):
        for v in vertices:
            if v.id == id:
                return v
        return None
    
class SpatialMultiVertex:
    def __init__(self, id, keywords):
        self.id = id
        self.keywords = keywords
        self.vertices = [SpatialVertex(str(id)+'-'+str(i), keyword) for i, keyword in enumerate(keywords)]
    def __str__(self):
        return str(self.id) + '(' + str(self.keywords) + ')'
    
class SpatialEdge:
    def __init__(self, id, vi, vj, lij = 0, uij = float('inf'), sign = '-', relation = None):
        # constraint should be a dict like {'lij':0, 'uij':1000, 'sign':'>', 'relation': disjoint}
        # 'sign' is always of of the four {'>', '<', '<>', '-'}
        # 'relation' should be a string, specifying the type of topological relation 
        # possible relations: intersects, contains, within, disjoint
        self.id = id
        self.vi = vi
        self.vj = vj
        if relation is not None and relation != 'disjoint':
            lij = 0
            uij = float('inf')
            sign = '-'
        self.constraint = {'lij': lij, 'uij': uij, 'sign': sign, 'relation': relation}
        self.constraint['is_exclusive'] = False if self.constraint['sign']=='-' else True
    def __str__(self):
        return str(self.id) + ': ' + str(self.vi) + ' ' + self.constraint['sign'] + ' ' + str(self.vj) + ' (' + str(self.constraint) + ')'

    def __eq__(self, another):
        return self.id == another.id and self.vi == another.vi and self.vj == another.vj and self.constraint['lij'] == another.constraint['lij'] and \
            self.constraint['uij'] == another.constraint['uij'] and self.constraint['sign'] == another.constraint['sign'] and \
            self.constraint['relation'] == another.constraint['relation']

    def __hash__(self):
        return hash(self.__str__())

    def get_constraint_label(self):
        label = ""
        lij, uij, relation = self.constraint['lij'], self.constraint['uij'], self.constraint['relation']
        if lij > 0 and uij < float('inf'):
            label += f"between {round(lij)} and {round(uij)}m\n"
        elif lij > 0:
            label += f"more than {round(lij)}m\n"
        elif uij < float('inf'):
            label += f"less than {round(uij)}m\n"
        if relation is not None:
            label += f"{self.constraint['relation']}\n"
        
        return label[:-1]

    def to_json(self):
        return json.dumps(self.to_dict())

    def to_dict(self):
        return {
                'id': self.id,
                'vi': self.vi.id,
                'vj': self.vj.id,
                'lij': self.constraint['lij'],
                'uij': self.constraint['uij'],
                'sign': self.constraint['sign'],
                'relation': self.constraint['relation']
        }

    @staticmethod
    def from_json(json_str, vertices):
        edge_dict = json.loads(json_str)
        return SpatialEdge.from_dict(edge_dict, vertices)

    @staticmethod
    def from_dict(edge_dict, vertices):
        id = edge_dict['id']
        vi = edge_dict['vi']
        vj = edge_dict['vj']
        lij = edge_dict['lij']
        uij = edge_dict['uij']
        sign = edge_dict['sign']
        relation = edge_dict['relation']
        vi = SpatialVertex.from_id(vi, vertices)
        vj = SpatialVertex.from_id(vj, vertices)
        return SpatialEdge(id, vi, vj, lij, uij, sign, relation)
    
    @staticmethod
    def get_edge_by_id(edges, id):
        for edge in edges:
            if edge.id == id:
                return edge
    
    
def find_edge(vertex_i, vertex_j, edges):
    for edge in edges:
        if edge.vi == vertex_i and edge.vj == vertex_j:
            return edge
    return None
    
class SpatialPatternMultiGraph:
    def __init__(self, multi_vertices, edges):
        # vertices should be a list of SpatialVertex objects 
        # edges should be a list of SpatialEdge objects
        self.pattern_type = 'Multi_keyword_vertices_graph'
        self.multi_vertices = multi_vertices
        self.edges = edges
        self.spatial_patterns = []
        keywords_of_vertices = [multi_vertex.keywords for multi_vertex in multi_vertices]
        for keywords_choice in cartesian_product(*keywords_of_vertices):
            simples_vertices = [SpatialVertex(i, wi) for i, wi in enumerate(keywords_choice)]
            simple_edges = []
            for i, multi_vertex_i in enumerate(multi_vertices):
                for j, multi_vertex_j in enumerate(multi_vertices):
                    edge_found = find_edge(multi_vertex_i, multi_vertex_j, edges)
                    if edge_found is not None:
                        lij, uij = edge_found.constraint['lij'], edge_found.constraint['uij']
                        sign, relation = edge_found.constraint['sign'], edge_found.constraint['relation']
                        simple_edges.append(SpatialEdge(str(i)+'-'+str(j), simples_vertices[i], simples_vertices[j], lij, uij, sign, relation))
        self.spatial_patterns.append(SpatialPatternGraph(simples_vertices, simple_edges))
        
    def __str__(self):
        descr = ""
        for edge in self.edges:
            descr += edge.__str__() + '\n'
        return descr
    
class SpatialPatternGraph:
    def __init__(self, vertices, edges):
        # vertices should be a list of SpatialVertex objects 
        # edges should be a list of SpatialEdge objects
        self.pattern_type = 'simple_graph'
        self.vertices = vertices
        self.edges = edges
        self.neighbors = defaultdict(list)
        self.pairs_to_edges = defaultdict(dict)
        for edge in edges:
            self.neighbors[edge.vi].append(edge.vj)
            self.neighbors[edge.vj].append(edge.vi)
            self.pairs_to_edges[edge.vi][edge.vj] = edge
            self.pairs_to_edges[edge.vj][edge.vi] = edge

    @staticmethod
    def from_json(json_str):
        sp_dict = json.loads(json_str)
        
        vertices = sp_dict['vertices']
        for i, vertex in enumerate(vertices):
            vertices[i] = SpatialVertex.from_dict(vertices[i])

        edges = sp_dict['edges']
        for i, edge in enumerate(edges):
            edges[i] = SpatialEdge.from_dict(edges[i], vertices)

        return SpatialPatternGraph(vertices, edges)

    def to_dict(self):
        ordered_vertices = sorted(self.vertices, key = lambda e: e.id)
        ordered_edges = sorted(self.edges, key = lambda e: e.id)
        sp_dict = {
            "vertices": [v.to_dict() for v in ordered_vertices],
            "edges": [e.to_dict() for e in ordered_edges]
        }
        return sp_dict

    def to_json(self, indent = None):
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)#.encode('utf8').decode()
    
    
    def __str__(self):
        descr = ""
        for edge in self.edges:
            descr += edge.__str__() + '\n'
        return descr
    
    def get_number_of_exclusion_contraints(self):
        number_of_exclusion_constraints = 0
        for edge in self.edges:
            if edge.constraint['sign'] == '>' or edge.constraint['sign'] == '<':
                number_of_exclusion_constraints += 1
            elif edge.constraint['sign'] == '<>':
                number_of_exclusion_constraints += 2
        return number_of_exclusion_constraints

    def __eq__(self, another):
        ordered_vertices = sorted(self.vertices, key = lambda e: e.id)
        ordered_edges = sorted(self.edges, key = lambda e: e.id)
        ordered_vertices_another = sorted(another.vertices, key = lambda e: e.id)
        ordered_edges_another = sorted(another.edges, key = lambda e: e.id)
        return len(ordered_vertices) == len(ordered_vertices_another) and \
                len(ordered_edges) == len(ordered_edges_another) and \
                all([(ordered_vertices[i] == ordered_vertices_another[i]) for i in range(len(ordered_vertices))]) and \
                all([(ordered_edges[i] == ordered_edges_another[i]) for i in range(len(ordered_edges))])

    def __hash__(self):
        return hash(self.to_json())

    def __lt__(self, another):
        return self.__hash__() < another.__hash__()
    
    def to_networkx(self):
        G = nx.Graph()
        for edge in self.edges:
            G.add_edge('   '+edge.vi.keyword, '   '+edge.vj.keyword, data = {'id': edge.id, 'constraint': edge.constraint})
        return G
    
    def plot_structure(self, output_file = None, positions = None, dpi = 85, node_color = 'k', edge_color = 'k', ax = None, 
             figwidth = 21, figheight = 7, xlim = (-1.15, 1.35), ylim = (-1.15, 1.15), node_size=400, edge_width=3.0):
        pattern_size = len(self.vertices)
        G = self.to_networkx()
        if positions is None:
            pos = nx.circular_layout(G)
        else:
            pos = {}
            for i,v in enumerate(self.vertices):
                pos['   '+v.keyword] = positions[i]

        #nx.draw(G, pos=nx.circular_layout(G), node_color=node_color, edge_color=edge_color)
        if ax is None:
            fig, ax = plt.subplots()
            fig.set_figwidth(figwidth)
            fig.set_figheight(figheight)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        nx.draw_networkx(G, pos=pos, ax = ax, with_labels=False, node_size=node_size, node_color=node_color, edge_color=edge_color)
        nx.draw_networkx_edges(G, pos=pos, width=edge_width, ax = ax)

        plt.tight_layout()
        if output_file is not None:
            plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
        # else:
        #     plt.show()

    
    def plot(self, output_file = None, positions = None, dpi = 85, node_color = np.array([[0.38431373, 0.61568627, 0.98823529]]), edge_color = 'k', ax = None, 
             figwidth = 21, figheight = 7, xlim = (-1.03, 1.35), ylim = (-1.05, 1.15), node_size=300, edge_width=1.0):
        #https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx_edge_labels.html#networkx.drawing.nx_pylab.draw_networkx_edge_labels
        #https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx_labels.html#networkx.drawing.nx_pylab.draw_networkx_labels
        #https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx_edges.html#networkx.drawing.nx_pylab.draw_networkx_edges
        #https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx_nodes.html#networkx.drawing.nx_pylab.draw_networkx_nodes
        #https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html
        
        font_sizes_by_pattern_size = {
            2: 28,
            3: 26,
            4: 24,
            5: 22,
            6: 20
        }
        pattern_size = len(self.vertices)
        G = self.to_networkx()
        if positions is None:
            pos = nx.circular_layout(G)
        else:
            pos = {}
            for i,v in enumerate(self.vertices):
                pos['   '+v.keyword] = positions[i]

        #nx.draw(G, pos=nx.circular_layout(G), node_color=node_color, edge_color=edge_color)
        if ax is None:
            fig, ax = plt.subplots()
            fig.set_figwidth(figwidth)
            fig.set_figheight(figheight)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        nx.draw_networkx(G, pos=pos, ax = ax, with_labels=False, node_color=node_color, node_size=node_size, edge_color=edge_color)
        nx.draw_networkx_edges(G, pos=pos, width=edge_width, ax = ax)
        nx.draw_networkx_labels(G, pos=pos, font_size = font_sizes_by_pattern_size[pattern_size], font_weight='bold', font_color = 'b', horizontalalignment='left', verticalalignment='bottom', ax=ax)

        edge_labels = {('   '+edge.vi.keyword, '   '+edge.vj.keyword): edge.get_constraint_label() for edge in self.edges}
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels = edge_labels, ax = ax, font_size=font_sizes_by_pattern_size[pattern_size])
        plt.tight_layout()
        if output_file is not None:
            plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
        else:
            plt.show()
    

    
def is_qq_e_match(ilq, os, ot, edge: SpatialEdge, metric):
    # this verification bellow is not necessary if the node matches are computed correctly
    # if not(edge.vi.keyword in os.keywords()) or not(edge.vj.keyword in ot.keywords()):
    #     return False
    
    lij, uij = edge.constraint['lij'], edge.constraint['uij']
    distance = os.distance(ot, metric)
    if not (lij <= distance <= uij):
        return False

    if edge.constraint['relation'] is not None and edge.constraint['relation'] != 'disjoint':
       if not bboxes_intersect(os.bbox(), ot.bbox()):
           return False
    # if edge.constraint['relation'] is not None and edge.constraint['relation'] != os.relation(ot):
    #     return False
        

    if edge.constraint['sign']=='>': #vi excludes vj
        # there should not be any object with vj's keyword nearer than lij from os
        circle_search_id = (edge.vj.keyword, os.centroid(), lij)# ((edge.vj.keyword,), os.centroid(), lij)
        if circle_search_id in ilq.cached_existence_searches:
            result = ilq.cached_existence_searches[circle_search_id]
        else:
            result = ilq.search_circle_existence(*circle_search_id)
            ilq.add_cached_existence_search(*circle_search_id, result)
            
        if result:
            return False
    elif edge.constraint['sign']=='<': #vj excludes vi
        # there should not be any object with vi's keyword nearer than lij from ot
        circle_search_id = (edge.vi.keyword, ot.centroid(), lij)
        if circle_search_id in ilq.cached_existence_searches:
            result = ilq.cached_existence_searches[circle_search_id]
        else:
            result = ilq.search_circle_existence(*circle_search_id)
            ilq.add_cached_existence_search(*circle_search_id, result)
            
        if result:
            return False
    elif edge.constraint['sign']=='<>': #vj mutual exclusion with vi
        # there should not be any object with vi's keyword nearer than lij from ot
        # and also, there should not be any object with vj's keyword nearer than lij from os
        circle_search_id = (edge.vj.keyword, os.centroid(), lij)
        if circle_search_id in ilq.cached_existence_searches:
            result = ilq.cached_existence_searches[circle_search_id]
        else:
            result = ilq.search_circle_existence(*circle_search_id)
            ilq.add_cached_existence_search(*circle_search_id, result)
            
        if result:
            return False
            
        circle_search_id = (edge.vi.keyword, ot.centroid(), lij)
        if circle_search_id in ilq.cached_existence_searches:
            result = ilq.cached_existence_searches[circle_search_id]
        else:
            result = ilq.search_circle_existence(*circle_search_id)
            ilq.add_cached_existence_search(*circle_search_id, result)
            
        if result:
            return False
    return True
            
    
def is_qq_n_match(node_i, node_j, edge: SpatialEdge, ilq, metric):
    # node_i e node_j are of type QuadNode
    bi = node_i.bbox
    bj = node_j.bbox
    lij = edge.constraint['lij']
    uij = edge.constraint['uij']

    if not (dmin(bi,bj,metric) <= uij and dmax(bi,bj,metric) >= lij):
        return False
    
    if edge.constraint['relation'] is not None:
        if edge.constraint['relation'] != 'disjoint' and not bboxes_intersect(bi, bj):
            return False
    
    if edge.constraint['sign'] == '-':
        return True
        
    elif edge.constraint['sign'] == '>':
        # we will do a radius search centered on the center point of node_i, and with radius max(0, lij-r(node_i))
        # r(node_i) represents the distance between the center of node_i and one of its extreme vertices.
        xci,yci = node_i.center
        xv,yv,_,_ = bi
        r_node_i = lat_lon_distance(yci, xci, yv, xv, metric)
        radius = max(0, lij - r_node_i)
        circle_search_id = (edge.vj.keyword, (xci,yci), radius)
        if circle_search_id in ilq.cached_existence_searches:
            result = ilq.cached_existence_searches[circle_search_id]
            #print('Reused')
        else:
            #print('search not cached')
            result = ilq.search_circle_existence(*circle_search_id)
            ilq.add_cached_existence_search(*circle_search_id, result)
        if result:
            return False
        return True
        
    elif edge.constraint['sign'] == '<':
        # we will do a radius search centered on the center point of node_j, and with radius max(0, lij-r(node_j))
        # r(node_j) represents the distance between the center of node_j and one of its extreme vertices.
        xcj,ycj = node_j.center
        xv,yv,_,_ = bj
        r_node_j = lat_lon_distance(ycj, xcj, yv, xv, metric)
        radius = max(0, lij - r_node_j)
        circle_search_id = (edge.vi.keyword, (xcj,ycj), radius)
        if circle_search_id in ilq.cached_existence_searches:
            result = ilq.cached_existence_searches[circle_search_id]
            #print('Reused')
        else:
            #print('search not cached')
            result = ilq.search_circle_existence(*circle_search_id)
            ilq.add_cached_existence_search(*circle_search_id, result)
        
        if result:
            return False
        return True
    else:
        xci,yci = node_i.center
        xv,yv,_,_ = bi
        r_node_i = lat_lon_distance(yci, xci, yv, xv, metric)
        radius = max(0, lij - r_node_i)
        circle_search_id_1 = (edge.vj.keyword, (xci,yci), radius)
        if circle_search_id_1 in ilq.cached_existence_searches:
            result1 = ilq.cached_existence_searches[circle_search_id_1]
        else:
            result1 = ilq.search_circle_existence(*circle_search_id_1)
            ilq.add_cached_existence_search(*circle_search_id_1, result1)
        if result1:
            return False
        
        xcj,ycj = node_j.center
        xv,yv,_,_ = bj
        r_node_j = lat_lon_distance(ycj, xcj, yv, xv, metric)
        radius = max(0, lij - r_node_j)
        circle_search_id_2 = (edge.vi.keyword, (xcj,ycj), radius)
        if circle_search_id_2 in ilq.cached_existence_searches:
            result2 = ilq.cached_existence_searches[circle_search_id_2]
        else:
            result2 = ilq.search_circle_existence(*circle_search_id_2)
            ilq.add_cached_existence_search(*circle_search_id_2, result2)
        if result2:
            return False
        return True


def find_sub_qq_n_matches(qq_n_match, candidate_nodes_vi, candidate_nodes_vj, edge, ilq, metric):
    qq_n_matches_l = []
    node_i, node_j = qq_n_match
    children_i = node_i.get_descendent_nodes_at_level(1)
    children_j = node_j.get_descendent_nodes_at_level(1)
    #print('teste0', len(children_i), len(children_j))
    if candidate_nodes_vi != set():
        # the intersection of children_i and candidate_nodes_vi
        children_i = list(filter(set(children_i).__contains__, candidate_nodes_vi))
        #print('teste1')
    if candidate_nodes_vj != set():
        children_j = list(filter(set(children_j).__contains__, candidate_nodes_vj))
        #print('teste2')
    for ci in children_i:
        for cj in children_j:
            if is_qq_n_match(ci, cj, edge, ilq, metric):
                qq_n_matches_l.append((ci,cj))
                #print(f'Teste/ edge {edge}\n qq_n_matches_l: {qq_n_matches_l}')
    return qq_n_matches_l


def compute_qq_n_matches_at_level_parallel(ilq: remote_ilquadtree.RemoteILQuadtree, edge: SpatialEdge, level: int, previous_qq_n_matches: list, metric, 
                                           candidate_nodes_vi = set(), candidate_nodes_vj = set(), pool_obj = None):
    find_sub_qq_n_matches_partial = partial(find_sub_qq_n_matches, candidate_nodes_vi = candidate_nodes_vi, 
                                    candidate_nodes_vj = candidate_nodes_vj, edge = edge, ilq = ilq, metric=metric)
    if pool_obj is not None:
        # parallel
        results = pool_obj.map(find_sub_qq_n_matches_partial, previous_qq_n_matches)
        qq_n_matches_l = list(itertools.chain(*results))
    else:
        qq_n_matches_l = []
        for qqnm in previous_qq_n_matches:
            qq_n_matches_l.extend(find_sub_qq_n_matches_partial(qqnm))

    return qq_n_matches_l        
                                                                       

def compute_qq_n_matches_for_all_edges(ilq: remote_ilquadtree.RemoteILQuadtree, sp: SpatialPatternGraph, metric, alternated, debug = False, pool_obj = None):
    #t0 = time()
    ilq.clean_caches()
    edges = sp.edges
    vertices = sp.vertices
    keywords = [v.keyword for v in vertices]
    depth = max([ilq.quadtrees[keyword].get_depth() for keyword in keywords])
    #qq_n_matches_by_level = {}
    #print('depth:', depth)
    # we need to reorder edges array to an optimal ordering to minimize computation efforts
    # 1) it partitions edges into two groups, where the first group
    # contains exclusive edges and the second group contains mutually
    # inclusive edges; 2) for each group, it ranks edges in an ascending
    # order of numbers of their n-matches in the previous level; and 3) by
    # concatenating edges in these two groups, it obtains the order of edges
    # for computing n-matches.
    exclusive_edges = [edge for edge in edges if edge.constraint['is_exclusive']]
    inclusive_edges = [edge for edge in edges if not edge.constraint['is_exclusive']]
    qq_n_matches_exclusive = dict()
    previous_qq_n_matches_exclusive = dict()
    qq_n_matches_inclusive = dict()
    previous_qq_n_matches_inclusive = dict()
    for ee in exclusive_edges:
        #print('exclusive edge:', ee)
        wi, wj = ee.vi.keyword, ee.vj.keyword
        previous_qq_n_matches_exclusive[ee] = [(ilq.quadtrees[wi].root, ilq.quadtrees[wj].root)]
    for ie in inclusive_edges:
        #print('inclusive edge:', ie)
        wi, wj = ie.vi.keyword, ie.vj.keyword
        previous_qq_n_matches_inclusive[ie] = [(ilq.quadtrees[wi].root, ilq.quadtrees[wj].root)]
    candidate_nodes = dict()
    #if len(edges) == 1:
    #    f_compute_qq_n_matches_at_level = compute_qq_n_matches_at_level
    #else:
    f_compute_qq_n_matches_at_level = compute_qq_n_matches_at_level_parallel
    for level in range(1, max(2,depth+1)):
        #print('level =', level)
        #print('Computing n-matches at level', level)
        for vertex in vertices:
            candidate_nodes[vertex] = set() # it is the set of nodes that are candidates to this vertex in this level
            
        for ee in exclusive_edges:
            #print('level, edge:', level, ee)
            qq_n_matches_exclusive[ee] = f_compute_qq_n_matches_at_level(ilq, ee, level, previous_qq_n_matches_exclusive[ee], metric, candidate_nodes[ee.vi], candidate_nodes[ee.vj], pool_obj = pool_obj)
            if debug:
                print(f'Total qq-n-matches for current edge {ee.id} at level {level}: {len(qq_n_matches_exclusive[ee])}')
            if len(qq_n_matches_exclusive[ee]) == 0:
                return 
            previous_qq_n_matches_exclusive[ee] = qq_n_matches_exclusive[ee]
            new_candidates_i, new_candidates_j = list(zip(*qq_n_matches_exclusive[ee]))
            if candidate_nodes[ee.vi]==set(): candidate_nodes[ee.vi] = set(new_candidates_i)
            else: candidate_nodes[ee.vi] = candidate_nodes[ee.vi].intersection(set(new_candidates_i))
            if candidate_nodes[ee.vj]==set(): candidate_nodes[ee.vj] = set(new_candidates_j)
            else: candidate_nodes[ee.vj] = candidate_nodes[ee.vj].intersection(set(new_candidates_j))
            
        for ie in inclusive_edges:
            #print('level, edge:', level, ie)
            qq_n_matches_inclusive[ie] = f_compute_qq_n_matches_at_level(ilq, ie, level, previous_qq_n_matches_inclusive[ie], metric, candidate_nodes[ie.vi], candidate_nodes[ie.vj], pool_obj = pool_obj)
            if debug:
                print(f'Total qq-n-matches for current edge {ie.id} at level {level}: {len(qq_n_matches_inclusive[ie])}')
            if len(qq_n_matches_inclusive[ie]) == 0:
                return 
            previous_qq_n_matches_inclusive[ie] = qq_n_matches_inclusive[ie]
            new_candidates_i, new_candidates_j = list(zip(*qq_n_matches_inclusive[ie]))
            if candidate_nodes[ie.vi]==set(): candidate_nodes[ie.vi] = set(new_candidates_i)
            else: candidate_nodes[ie.vi] = candidate_nodes[ie.vi].intersection(set(new_candidates_i))
            if candidate_nodes[ie.vj]==set(): candidate_nodes[ie.vj] = set(new_candidates_j)
            else: candidate_nodes[ie.vj] = candidate_nodes[ie.vj].intersection(set(new_candidates_j))
            
        # sort list  exclusive_edges according to len(qq_n_matches_exclusive[ee])
        # also sort the list inclusive_edges according to len(qq_n_matches_inclusive[ie])
        #qq_n_matches_by_level[level] = {**qq_n_matches_exclusive.copy(), **qq_n_matches_inclusive.copy()}
        exclusive_edges.sort(key = lambda ee: len(qq_n_matches_exclusive[ee]))
        inclusive_edges.sort(key = lambda ie: len(qq_n_matches_inclusive[ie]))
    

    return {**qq_n_matches_exclusive, **qq_n_matches_inclusive}


def compute_qq_n_matches_for_all_edges_alternated(ilq: remote_ilquadtree.RemoteILQuadtree, sp: SpatialPatternGraph, metric, alternated, debug = False, pool_obj = None):
    #t0 = time()
    ilq.clean_caches()
    edges = sp.edges
    vertices = sp.vertices
    keywords = [v.keyword for v in vertices]
    depth = max([ilq.quadtrees[keyword].get_depth() for keyword in keywords])
    #qq_n_matches_by_level = {}
    #print('depth:', depth)
    # we need to reorder edges array to an optimal ordering to minimize computation efforts
    # 1) it partitions edges into two groups, where the first group
    # contains exclusive edges and the second group contains mutually
    # inclusive edges; 2) for each group, it ranks edges in an ascending
    # order of numbers of their n-matches in the previous level; and 3) by
    # concatenating edges in these two groups, it obtains the order of edges
    # for computing n-matches.
    qq_n_matches = dict()
    previous_qq_n_matches = dict()
    for ee in edges:
        #print('exclusive edge:', ee)
        wi, wj = ee.vi.keyword, ee.vj.keyword
        previous_qq_n_matches[ee] = [(ilq.quadtrees[wi].root, ilq.quadtrees[wj].root)]
    candidate_nodes = dict()
    
    edges = get_edges_order({ee: not (ee.constraint['is_exclusive']) for ee in edges},edges,sp,alternated=alternated)
    f_compute_qq_n_matches_at_level = compute_qq_n_matches_at_level_parallel
    for level in range(1, max(2,depth+1)):
        #print('level =', level)
        #print('Computing n-matches at level', level)
        for vertex in vertices:
            candidate_nodes[vertex] = set() # it is the set of nodes that are candidates to this vertex in this level
            
        for ee in edges:
            #print('level, edge:', level, ee)
            qq_n_matches[ee] = f_compute_qq_n_matches_at_level(ilq, ee, level, previous_qq_n_matches[ee], metric, candidate_nodes[ee.vi], candidate_nodes[ee.vj], pool_obj = pool_obj)
            if debug:
                print(f'Total qq-n-matches for current edge {ee.id} at level {level}: {len(qq_n_matches[ee])}')
            if len(qq_n_matches[ee]) == 0:
                return 
            previous_qq_n_matches[ee] = qq_n_matches[ee]
            new_candidates_i, new_candidates_j = list(zip(*qq_n_matches[ee]))
            if candidate_nodes[ee.vi]==set(): candidate_nodes[ee.vi] = set(new_candidates_i)
            else: candidate_nodes[ee.vi] = candidate_nodes[ee.vi].intersection(set(new_candidates_i))
            if candidate_nodes[ee.vj]==set(): candidate_nodes[ee.vj] = set(new_candidates_j)
            else: candidate_nodes[ee.vj] = candidate_nodes[ee.vj].intersection(set(new_candidates_j))
            
        # sort list  exclusive_edges according to len(qq_n_matches_exclusive[ee])
        # also sort the list inclusive_edges according to len(qq_n_matches_inclusive[ie])
        #qq_n_matches_by_level[level] = {**qq_n_matches_exclusive.copy(), **qq_n_matches_inclusive.copy()}
        edges = get_edges_order({ee: len(qq_n_matches[ee]) for ee in edges},edges,sp,alternated=alternated)
    

    return qq_n_matches


def compute_qq_n_matches_for_all_levels(ilq: remote_ilquadtree.RemoteILQuadtree, sp: SpatialPatternGraph, metric, debug = False, pool_obj = None):
    #t0 = time()
    if pool_obj is None:
        pool_obj = ThreadPool(int(multiprocessing.cpu_count()-1)) #ThreadPool
    edges = sp.edges
    vertices = sp.vertices
    keywords = [v.keyword for v in vertices]
    depth = max([ilq.quadtrees[keyword].get_depth() for keyword in keywords])
    #qq_n_matches_by_level = {}
    #print('depth:', depth)
    # we need to reorder edges array to an optimal ordering to minimize computation efforts
    # 1) it partitions edges into two groups, where the first group
    # contains exclusive edges and the second group contains mutually
    # inclusive edges; 2) for each group, it ranks edges in an ascending
    # order of numbers of their n-matches in the previous level; and 3) by
    # concatenating edges in these two groups, it obtains the order of edges
    # for computing n-matches.
    exclusive_edges = [edge for edge in edges if edge.constraint['is_exclusive']]
    inclusive_edges = [edge for edge in edges if not edge.constraint['is_exclusive']]
    qq_n_matches_exclusive = dict()
    previous_qq_n_matches_exclusive = dict()
    qq_n_matches_inclusive = dict()
    previous_qq_n_matches_inclusive = dict()
    for ee in exclusive_edges:
        #print('exclusive edge:', ee)
        wi, wj = ee.vi.keyword, ee.vj.keyword
        previous_qq_n_matches_exclusive[ee] = [(ilq.quadtrees[wi].root, ilq.quadtrees[wj].root)]
    for ie in inclusive_edges:
        #print('inclusive edge:', ie)
        wi, wj = ie.vi.keyword, ie.vj.keyword
        previous_qq_n_matches_inclusive[ie] = [(ilq.quadtrees[wi].root, ilq.quadtrees[wj].root)]
    candidate_nodes = dict()
    qq_n_matches_levels = defaultdict(dict)
    qq_n_matches_levels[0] = {**previous_qq_n_matches_exclusive, **previous_qq_n_matches_inclusive}
    #if len(edges) == 1:
    #    f_compute_qq_n_matches_at_level = compute_qq_n_matches_at_level
    #else:
    f_compute_qq_n_matches_at_level = compute_qq_n_matches_at_level_parallel
    for level in range(1, max(2,depth+1)):
        #print('level =', level)
        #print('Computing n-matches at level', level)
        for vertex in vertices:
            candidate_nodes[vertex] = set() # it is the set of nodes that are candidates to this vertex in this level
            
        for ee in exclusive_edges:
            #print('level, edge:', level, ee)
            qq_n_matches_exclusive[ee] = f_compute_qq_n_matches_at_level(ilq, ee, level, previous_qq_n_matches_exclusive[ee], metric, candidate_nodes[ee.vi], candidate_nodes[ee.vj], pool_obj = pool_obj)
            if debug:
                print(f'Total qq-n-matches for current edge {ee.id} at level {level}: {len(qq_n_matches_exclusive[ee])}')
            if len(qq_n_matches_exclusive[ee]) == 0:
                return 
            previous_qq_n_matches_exclusive[ee] = qq_n_matches_exclusive[ee]
            new_candidates_i, new_candidates_j = list(zip(*qq_n_matches_exclusive[ee]))
            if candidate_nodes[ee.vi]==set(): candidate_nodes[ee.vi] = set(new_candidates_i)
            else: candidate_nodes[ee.vi] = candidate_nodes[ee.vi].intersection(set(new_candidates_i))
            if candidate_nodes[ee.vj]==set(): candidate_nodes[ee.vj] = set(new_candidates_j)
            else: candidate_nodes[ee.vj] = candidate_nodes[ee.vj].intersection(set(new_candidates_j))
            
        for ie in inclusive_edges:
            #print('level, edge:', level, ie)
            qq_n_matches_inclusive[ie] = f_compute_qq_n_matches_at_level(ilq, ie, level, previous_qq_n_matches_inclusive[ie], metric, candidate_nodes[ie.vi], candidate_nodes[ie.vj], pool_obj = pool_obj)
            if debug:
                print(f'Total qq-n-matches for current edge {ie.id} at level {level}: {len(qq_n_matches_inclusive[ie])}')
            if len(qq_n_matches_inclusive[ie]) == 0:
                return 
            previous_qq_n_matches_inclusive[ie] = qq_n_matches_inclusive[ie]
            new_candidates_i, new_candidates_j = list(zip(*qq_n_matches_inclusive[ie]))
            if candidate_nodes[ie.vi]==set(): candidate_nodes[ie.vi] = set(new_candidates_i)
            else: candidate_nodes[ie.vi] = candidate_nodes[ie.vi].intersection(set(new_candidates_i))
            if candidate_nodes[ie.vj]==set(): candidate_nodes[ie.vj] = set(new_candidates_j)
            else: candidate_nodes[ie.vj] = candidate_nodes[ie.vj].intersection(set(new_candidates_j))
            
        # sort list  exclusive_edges according to len(qq_n_matches_exclusive[ee])
        # also sort the list inclusive_edges according to len(qq_n_matches_inclusive[ie])
        #qq_n_matches_by_level[level] = {**qq_n_matches_exclusive.copy(), **qq_n_matches_inclusive.copy()}
        exclusive_edges.sort(key = lambda ee: len(qq_n_matches_exclusive[ee]))
        inclusive_edges.sort(key = lambda ie: len(qq_n_matches_inclusive[ie]))

        qq_n_matches_levels[level] = {**qq_n_matches_exclusive, **qq_n_matches_inclusive}
    

    return qq_n_matches_levels


def is_connected(vertex, vertices, edges):
    vertices_pairs = [(edge.vi, edge.vj) for edge in edges]
    vertices_pairs = list(filter(lambda vp: vertex in vp, vertices_pairs))
    for vp in vertices_pairs:
        if vp[0]==vertex and vp[1] in vertices:
            return True
        if vp[1]==vertex and vp[0] in vertices:
            return True
    return False

def find_skip_edges(edges_order):
    connected_vertices_subgraphs = []
    skip_edges = []
    for edge in edges_order:
        if not edge.constraint['is_exclusive']:
            for vertices_subgraph in connected_vertices_subgraphs:
                if {edge.vi, edge.vj}.issubset(vertices_subgraph):
                    skip_edges.append(edge)
                    break
        if connected_vertices_subgraphs==[]:
            connected_vertices_subgraphs.append({edge.vi, edge.vj})
        else:
            found_connected_subgraph = False
            for i,vertices_subgraph in enumerate(connected_vertices_subgraphs):
                # find the subgraph that is connected (by some edge) to vi or vj, if there is any
                # if not, create a new subgraph for that edge
                if is_connected(edge.vi, vertices_subgraph, edges_order) or \
                    is_connected(edge.vj, vertices_subgraph, edges_order):
                    connected_vertices_subgraphs[i].add(edge.vi)
                    connected_vertices_subgraphs[i].add(edge.vj)
                    found_connected_subgraph = True
                    break
            # if connected_vertices_subgraphs wasn't empty but didn't have a connected subgraph to this edge, create a new subgraph
            if not found_connected_subgraph:
                connected_vertices_subgraphs.append({edge.vi, edge.vj})
    return skip_edges        


def find_sub_qq_e_matches(qq_n_match, edge, ilq, candidate_objects, metric):
    #print('started running find_sub_qq_e_matches')
    #t0 = time()
    qq_e_matches = []
    node_i,node_j = qq_n_match
    oss = node_i.get_objects()
    ots = node_j.get_objects()
    candidate_objects_vi = candidate_objects[edge.vi]
    candidate_objects_vj = candidate_objects[edge.vj]

    if candidate_objects_vi != set():
        # the intersection of children_i and candidate_nodes_vi
        oss = list(filter(set(candidate_objects_vi).__contains__, oss))
    if candidate_objects_vj != set():
        ots = list(filter(set(candidate_objects_vj).__contains__, ots))
    #print('Total oss, ots pairs:', len(oss), len(ots))
    for os in oss:
        for ot in ots:
            if is_qq_e_match(ilq, os, ot, edge, metric):
                qq_e_matches.append((os,ot))
    #print('time spent on running find_sub_qq_e_matches:', time()-t0)
    #print('ended running find_sub_qq_e_matches')
    return qq_e_matches

def compute_qq_e_matches_for_an_edge_parallel(ilq, edge, qq_n_matches_for_the_edge, metric, candidate_objects = dict(), 
            pool_obj = None):
    find_sub_qq_e_matches_partial = partial(find_sub_qq_e_matches, edge = edge, ilq = ilq, candidate_objects = candidate_objects, 
                                                metric=metric)
    if pool_obj is not None:
        results = pool_obj.map(find_sub_qq_e_matches_partial, qq_n_matches_for_the_edge)
        qq_e_matches = set(itertools.chain(*results))
    else:
        #sequential
        qq_e_matches = []
        for n_mt in qq_n_matches_for_the_edge:
            qq_e_matches.extend(find_sub_qq_e_matches_partial(n_mt))
        qq_e_matches = set(qq_e_matches)

    return qq_e_matches

def compute_qq_e_matches_for_all_edges(ilq: remote_ilquadtree.RemoteILQuadtree, sp: SpatialPatternGraph, qq_n_matches: dict, metric, alternated, debug = True, pool_obj = None):
    #t0 = time()
    edges = sp.edges
    vertices = sp.vertices
    # we need to reorder edges array according to qq_n_matches dictionary
    #edges_order = sorted(edges, key = lambda e: len(qq_n_matches[e]) or 0)
    edges_values_dict = {e: len(qq_n_matches[e]) or 0 for e in edges}
    if debug:
        print('Reordering edges by qq-n-matches for filtering the non-skip edges')
    edges_order = get_edges_order(edges_values_dict, edges, sp, alternated = False, debug=debug)
    skip_edges = find_skip_edges(edges_order)
    non_skip_edges = [e for e in edges if e not in skip_edges]
    non_skip_edges = get_edges_order(edges_values_dict, non_skip_edges, sp, alternated=alternated)
    qq_e_matches = dict()
    
    candidate_objects = {vertex: set() for vertex in vertices} # it saves the set of objects that are candidates to each vertex 
    for edge in non_skip_edges:
        qq_e_matches[edge] = compute_qq_e_matches_for_an_edge_parallel(ilq, edge, qq_n_matches[edge], metric, candidate_objects, pool_obj = pool_obj)
        if debug:
            print(f'- Total qq-e-matches for edge {edge.id}: {len(qq_e_matches[edge])}')
        if len(qq_e_matches[edge])==0:
            return None, skip_edges, non_skip_edges
        candidate_objects_i, candidate_objects_j = list(zip(*qq_e_matches[edge]))
        if candidate_objects[edge.vi]==set(): candidate_objects[edge.vi] = set(candidate_objects_i)
        else: candidate_objects[edge.vi] = candidate_objects[edge.vi].intersection(set(candidate_objects_i))
        if candidate_objects[edge.vj]==set(): candidate_objects[edge.vj] = set(candidate_objects_j)
        else: candidate_objects[edge.vj] = candidate_objects[edge.vj].intersection(set(candidate_objects_j))

    return qq_e_matches, skip_edges, non_skip_edges

def generate_partial_solution_from_qq_e_match(qq_e_match, edge, sp):
    os, ot = qq_e_match
    partial_solution = {vertex.id: None for vertex in sp.vertices} #hashabledict()
    # for vertex in sp.vertices:
    #     partial_solution[vertex.id] = None
    partial_solution[edge.vi.id] = os
    partial_solution[edge.vj.id] = ot
    return partial_solution

def generate_partial_solutions_from_qq_e_matches(qq_e_matches, edge, sp, pool_obj = None):
    generate_partial_solution_from_qq_e_match_partial = partial(generate_partial_solution_from_qq_e_match, edge = edge, sp = sp)
    if pool_obj is not None:
        partial_solutions = pool_obj.map(generate_partial_solution_from_qq_e_match_partial, qq_e_matches)
    else:
        partial_solutions = [generate_partial_solution_from_qq_e_match_partial(qqem) for qqem in qq_e_matches]
    return partial_solutions

def merge_partial_solutions(pa, pb, sp):
    # pa and pb and dictionaries in the format: {v1: obj1, ..., vn: objn} where vi's are vertices and obji's are GeoObjs
    # merging means aggregating the two partial solutions into a single one if possible
    # sometimes it's not possible, when the two solutions provide a different value for the same vertex
    merged = dict()# hashabledict
    for vertex in sp.vertices:
        if pa[vertex.id] is not None and pb[vertex.id] is not None and pa[vertex.id]!=pb[vertex.id]:
            return None # there is no merge (merging is impossible)
        merged[vertex.id] = pa[vertex.id] or pb[vertex.id] # becomes the one that is not the 'None' if there is one not being None
    return merged


def merge_lists_of_partial_solutions(pas, pbs, sp):
    merges_list = []

    for pa in pas:
        for pb in pbs:
            merge = merge_partial_solutions(pa, pb, sp)
            if merge is not None:
                merges_list.append(merge)
    return merges_list

def merge_lists_of_partial_solutions_parallel(pas, pbs, sp, pool_obj):
    merge_pa_with_pbs_partial = partial(merge_pa_with_pbs, pbs=pbs, sp=sp)
    merges = list(itertools.chain(*pool_obj.map(merge_pa_with_pbs_partial, pas)))
    return merges

def merge_indexes_of_partial_solutions(indices_pair, pas, pbs, sp):
    (i,j) = indices_pair
    return merge_partial_solutions(pas[i], pbs[j], sp)

def merge_pa_with_pbs(pa, pbs, sp):
    merges = []
    for pb in pbs:
        merge = merge_partial_solutions(pa, pb, sp)
        if merge is not None:
            merges.append(merge)
    return merges




def filter_qq_e_matches_by_vertex_candidates(qq_e_matches, edge, candidates):
    #return [e for e in qq_e_matches if (e[0] in candidates[edge.vi] and e[1] in candidates[edge.vj])]
    return list(filter(lambda e: (e[0] in candidates[edge.vi] and e[1] in candidates[edge.vj]), qq_e_matches))
    

def join_qq_e_matches(sp: SpatialPatternGraph, qq_e_matches: dict, qq_n_matches: dict, skip_edges: list, non_skip_edges: list, metric, 
                      alternated, pool_obj = None, debug = False):
    #t0 = time()
    #non_skip_edges.sort(key = lambda e: len(qq_e_matches[e]))
    #skip_edges.sort(key = lambda e: len(qq_n_matches[e]))

    if debug:
        print('Non-skip edges order before candidate objects filtering:', [e.id for e in non_skip_edges])
    #ordered_edges = non_skip_edges + skip_edges
    #vertices = sp.vertices

    if debug:
        print('Constructing vertex candidate objects for Join')
    candidates = {vertex: set() for vertex in sp.vertices}
    for edge in non_skip_edges:
        cvi, cvj = list(zip(*qq_e_matches[edge]))
        if candidates[edge.vi] == set(): candidates[edge.vi] = set(cvi)
        else: candidates[edge.vi] = candidates[edge.vi].intersection(set(cvi))
        if candidates[edge.vj] == set(): candidates[edge.vj] = set(cvj)
        else: candidates[edge.vj] = candidates[edge.vj].intersection(set(cvj))
    if debug:
        print('Total candidate objects by vertex:', {v: len(candidates[v]) for v in candidates})

    if debug:
        print('Filtering qq-e-matches by vertex candidate objects for Join')
    for edge in non_skip_edges:
        qq_e_matches[edge] = filter_qq_e_matches_by_vertex_candidates(qq_e_matches[edge], edge, candidates)

    if debug:
        print('Sorting edges by the alternating total qq-e-matches estrategy for Join')

    non_skip_edges.sort(key = lambda edge: len(qq_e_matches[edge]))

    if debug:
        print('Non-skip edges order after candidate objects filtering:', [{e.id: len(qq_e_matches[e])} for e in non_skip_edges])
    if len(non_skip_edges) >= 3 and alternated:
        edges_values_dict = {e: len(qq_e_matches[e]) for e in non_skip_edges}
        non_skip_edges = get_edges_order(edges_values_dict, non_skip_edges, sp, alternated=alternated, debug=debug)
    if debug:
        print('Non-skip edges after alternated ordering:', [e.id for e in non_skip_edges])


    if debug:
        print('Generating partial solutions')
    partial_solutions = [{vertex.id: None for vertex in sp.vertices}]
    for edge in non_skip_edges:
        if debug:
            print(f'Generating partial solutions from qq-e-matches of edge {edge.id}')
        partial_solutions_edge = generate_partial_solutions_from_qq_e_matches(qq_e_matches[edge], edge, sp, pool_obj = pool_obj)
        if debug:
            print(f'Joining partial solutions of edge {edge.id} with the accumulated partial solutions of the previously processed edges')
            print(f'len(partial_solutions_edge): {len(partial_solutions_edge)}/ len(partial_solutions): {len(partial_solutions)}')
        partial_solutions = merge_lists_of_partial_solutions(partial_solutions, partial_solutions_edge, sp)

    if debug:
        print('Filtering partial solutions by skip edges constraints and connectivity constraints')
    if pool_obj is not None:
        partial_solution_satisfies_skips_and_connectivities_partial = partial(partial_solution_satisfies_skip_edges_and_connectivity, 
                            skip_edges=skip_edges, metric=metric, edges=sp.edges)
        final_indexes = np.where(np.array(pool_obj.map(partial_solution_satisfies_skips_and_connectivities_partial, partial_solutions)))[0]#.tolist()
        final_solutions = np.array(partial_solutions)[final_indexes]
    else:
        for edge in skip_edges:
            for i,solution in enumerate(partial_solutions):
                if solution is None:
                    continue
                vi, vj = edge.vi, edge.vj
                lij, uij = edge.constraint['lij'], edge.constraint['uij']
                os, ot = solution[vi.id], solution[vj.id]
                distance_ = os.distance(ot, metric)
                if not(lij <= distance_ <= uij):
                    partial_solutions[i] = None
        partial_solutions = filter(lambda x: x is not None, partial_solutions)
        final_solutions = []
        for solution in partial_solutions:
            if partial_solution_satisfies_connectivities(solution, sp.edges): # and partial_solution_satisfies_skip_edges(solution, skip_edges, metric)
                final_solutions.append(solution)

    return final_solutions

def partial_solution_satisfies_skip_edges_and_connectivity(solution, skip_edges, metric, edges):
    return partial_solution_satisfies_skip_edges(solution, skip_edges, metric) and \
        partial_solution_satisfies_connectivities(solution, edges)


def partial_solution_satisfies_skip_edges(solution, skip_edges, metric):
    for edge in skip_edges:
        if solution is None:
            continue
        vi, vj = edge.vi, edge.vj
        lij, uij = edge.constraint['lij'], edge.constraint['uij']
        os, ot = solution[vi.id], solution[vj.id]
        distance_ = os.distance(ot, metric)
        if not(lij <= distance_ <= uij):
            return False
    return True

def partial_solution_satisfies_connectivities(solution, edges):
    for edge in edges:
        relation = edge.constraint['relation']
        if relation is not None:
            vi, vj = edge.vi, edge.vj
            os, ot = solution[vi.id], solution[vj.id]
            #if edge.constraint['relation'] not in os.relations_with(ot):
            if (relation == 'intersects' and not os.item['geometry'].intersects(ot.item['geometry'])) or \
                (relation == 'contains' and not os.item['geometry'].contains(ot.item['geometry'])) or \
                (relation == 'within' and not ot.item['geometry'].contains(os.item['geometry'])) or \
                (relation == 'disjoint' and os.item['geometry'].intersects(ot.item['geometry'])):
                return False
    return True

def get_edges_of_vertex(vertex, sp):
    return list(sp.pairs_to_edges[vertex].values())

def get_edges_order(edges_values_dict, initial_edges_order, sp, alternated: bool, debug = False):
    if debug:
        print(f'Edges and values dict: {[(e.id, v) for e,v in edges_values_dict.items()]}')
    edges_sequence = []
    remaining_edges = set(initial_edges_order)
    reverse = False # if false searches the min, if True, searches the max
    current_edge = sorted(initial_edges_order, key = lambda e: edges_values_dict[e], reverse=reverse)[0]
    if debug:
        print('Starting edge of sequence:', current_edge.id)
    if alternated:
        reverse = not reverse
    current_vertex = current_edge.vj
    edges_sequence.append(current_edge)
    remaining_edges -= {current_edge}
    visited_vertices_with_remaining_edges = set()
    if len(set(get_edges_of_vertex(current_edge.vi, sp)).intersection(set(initial_edges_order)) - set(edges_sequence))>0:
        visited_vertices_with_remaining_edges.add(current_edge.vi)
    while len(remaining_edges) > 0:
        candidates_next_edge = set(get_edges_of_vertex(current_vertex, sp)).intersection(set(initial_edges_order)) - set(edges_sequence)
                
        if len(candidates_next_edge) > 0:
            visited_vertices_with_remaining_edges.add(current_vertex)
            next_edge = sorted(candidates_next_edge, key = lambda e: edges_values_dict[e], reverse=reverse)[0]
            if alternated:
                reverse = not reverse
            if debug:
                print('Continued path - Next edge of sequence:', next_edge.id)
            if current_vertex == next_edge.vi:
                current_vertex = next_edge.vj
            else:
                current_vertex = next_edge.vi
            edges_sequence.append(next_edge)
            remaining_edges -= {next_edge} # (current_vertex, next_vertex), (next_vertex, current_vertex)
        else:
            visited_vertices_with_remaining_edges -= {current_vertex}
            if len(visited_vertices_with_remaining_edges) == 0:
                if debug:
                    print('No more visited vertices with remaining edges are left. Returning edges sequence:', [e.id for e in edges_sequence])
                return edges_sequence
            next_edge = sorted(list(filter(lambda e: ((e.vi in visited_vertices_with_remaining_edges)or(e.vj in visited_vertices_with_remaining_edges)), 
                    remaining_edges)), key = lambda e: edges_values_dict[e], reverse=reverse)[0]
            if alternated:
                reverse = not reverse
            if debug:
                print('Changed path - Next edge of sequence:', next_edge.id)
            edges_sequence.append(next_edge)
            remaining_edges -= {next_edge}
            if next_edge.vi in visited_vertices_with_remaining_edges:
                current_vertex = next_edge.vj
            else:
                current_vertex = next_edge.vi
        visited_vertices_with_remaining_edges = set(filter(lambda v: len(set(get_edges_of_vertex(v, sp)).intersection(set(initial_edges_order)) - set(edges_sequence))>0 , 
                                                              visited_vertices_with_remaining_edges))
    if debug:
        print('No more remaing edges. Returning edges sequence:', [e.id for e in edges_sequence])
    return edges_sequence

def QQESPM(sp, ilquadtree: remote_ilquadtree.RemoteILQuadtree = None, metric = 'geodesic', ilq_dir = ilq_object_path, 
        ilq_get_method = 'load', 
        data_dir = os.path.dirname(os.path.realpath(__file__)) + '/data/pois_paraiba5.csv', parallel=False, alternated=True, debug = False):
    global remote_ilqs
    global current_ilq_dir
    global remote_ilq

    for ilq in remote_ilqs:
        if remote_ilqs[ilq] is not None:
            remote_ilqs[ilq].clean_caches()
            remote_ilqs[ilq].balance_memory_allocation()
    
    if ilquadtree is not None:
        remote_ilq = ilquadtree

    elif current_ilq_dir != ilq_dir:
        current_ilq_dir = ilq_dir
        if ilq_get_method == 'load':
            if debug:
                print('Loading remote ILQuadtree ...')
            if ilq_dir in remote_ilqs:
                remote_ilq = remote_ilqs[ilq_dir]
            else:
                remote_ilq = load_remote_ilquadtree(ilq_dir)#generate_remote_ilquadtree()
                remote_ilqs[ilq_dir] = remote_ilq
        elif ilq_get_method == 'generate':
            if debug:
                print('Generating remote ILQuadtree ...')
            remote_ilq = generate_remote_ilquadtree(data_dir)
    t0 = time()

    if parallel:
        pool_obj = ThreadPool(max(2, int(multiprocessing.cpu_count()-3))) #ThreadPool
    else:
        pool_obj = None
    
    keywords = [v.keyword for v in sp.vertices]
    if any([keyword not in remote_ilq.quadtrees for keyword in keywords]):
        if debug:
            missing_keyword = keywords[[keyword not in remote_ilq.quadtrees for keyword in keywords].index(True)]
            print(f'Zero solutions, since keyword "{missing_keyword}" is not present in the dataset')
        return [], time() - t0, psutil.Process().memory_info().rss/(2**20)
    if debug:
        print('Computing qq-n-matches for edges')
    qq_n_matches = compute_qq_n_matches_for_all_edges(remote_ilq, sp, metric, alternated=alternated, debug = debug, pool_obj = pool_obj)
    if qq_n_matches is None:
        return [], time() - t0, psutil.Process().memory_info().rss/(2**20)
    if debug:
        for edge in qq_n_matches:
            print(f'- Total qq-n-matches for edge {edge.id}: {len(qq_n_matches[edge])}')
        print('Computing qq-e-matches for edges')
    
    qq_e_matches, skip_edges, non_skip_edges = compute_qq_e_matches_for_all_edges(remote_ilq, sp, qq_n_matches, metric, alternated=alternated, debug=debug, pool_obj = pool_obj)
    if debug:
        print('Skip edges:', [e.id for e in skip_edges])
        print('Non-skip edges:', [e.id for e in non_skip_edges])
    if qq_e_matches is None:
        return [], time() - t0, psutil.Process().memory_info().rss/(2**20)
    else:
        if debug:
            print('Number of skip-edges:', len(skip_edges))
            print('Joining qq-e-matches')
        solutions = join_qq_e_matches(sp, qq_e_matches, qq_n_matches, skip_edges, non_skip_edges, metric, 
                                      alternated=alternated, pool_obj = pool_obj, debug = debug)
        # solutions is a list of dictionaries in the format {v1: obj1, v2: obj2, ..., vn: objn} with matches to vertices 
    memory_usage = psutil.Process().memory_info().rss/(2**20)
    if pool_obj is not None:
        pool_obj.close()
    elapsed_time = time() - t0
    return solutions, elapsed_time, memory_usage
            
        
def qqespm_find_solutions(ilquadtree, pattern):
    solutions, _, _ = QQESPM(ilquadtree, pattern)
    return solutions

def solutions_to_json(solutions, indent=None, only_ids = False):
    solutions_json_list = []
    for solution in solutions:
        if only_ids:
            solutions_json_list.append({vertex_id: solution[vertex_id].item.get('osm_id') for vertex_id in solution})
        else:
            solutions_json_list.append({vertex_id: solution[vertex_id].get_data() for vertex_id in solution})
    solutions_dict = {'solutions': solutions_json_list}
    return json.dumps(solutions_dict, indent=indent, ensure_ascii=False).encode('utf8').decode()




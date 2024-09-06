from elasticsearch import Elasticsearch
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from time import time
import psutil
from shapely.geometry import shape
import json
from multiprocessing.pool import ThreadPool
import multiprocessing
from functools import partial
import resource_monitoring

#default_index = "paraiba-pois-osm-index5"
current_index_name = None
es = None

class hashabledict(dict):
    def __hash__(self):
        return hash(json.dumps(self))

def hits_are_equal(hit_i: hashabledict, hit_j: hashabledict):
    return hit_i['fields']['properties.osm_id'][0] == hit_j['fields']['properties.osm_id'][0]

def update_candidate_objects(candidate_objects, edge, qq_e_matches_edge):
    candidate_vi, candidate_vj = list(zip(*qq_e_matches_edge))
    #candidate_vi, candidate_vj = list(candidate_vi), list(candidate_vj)
    # for i, candidate in enumerate(candidate_vi):
    #     candidate_vi[i] = hashabledict(**candidate)
    # for i, candidate in enumerate(candidate_vj):
    #     candidate_vj[i] = hashabledict(**candidate)
    candidate_vi, candidate_vj = set(candidate_vi), set(candidate_vj)
    new_candidate_objects = {edge.vi.keyword: candidate_vi, edge.vj.keyword: candidate_vj}
    for keyword in new_candidate_objects:
        if keyword in candidate_objects:
            ids_new_candidate_objects = [c['fields']['properties.osm_id'][0] for c in new_candidate_objects[keyword]]
            candidate_objects[keyword] = set(filter(lambda c: c['fields']['properties.osm_id'][0] in ids_new_candidate_objects, candidate_objects[keyword]))
            #candidate_objects[keyword] = candidate_objects[keyword].intersection(new_candidate_objects[keyword])
        else:
            candidate_objects[keyword] = new_candidate_objects[keyword]

def establish_elastic_connection(config_dir = 'config', method='https', debug = False):
    if debug:
        print('Establishing Elasticsearch connection ...')
    ELASTIC_PASSWORD = open(f'{config_dir}/elasticpassword').read()

    es = Elasticsearch(
        [f'{method}://localhost:9200'],
        request_timeout=1000,
        basic_auth=('elastic', ELASTIC_PASSWORD),
        verify_certs=False,
        ssl_show_warn = False
    )
    if debug:
        print(es.info())
    return es

def search_by_internal_id(es, index_name, _id):
    return es.get(index=index_name, id=_id)
    
def search_by_osm_id(es, index_name, osm_id, size = 100000, count = False):
    query = {
      "query": {
        "match": {
          "properties.osm_id": osm_id
        }
      }
    }
    if count:
        return es.count(index = index_name, body = query)['count']
    else:
        #query["fields"] = ["geometry", "properties.osm_id"]
        query["_source"] = True
        query["size"] = size
        return es.search(index = index_name, body = query)
    
def get_all_osm_ids_from_index(es, index_name, max_size=130000):
    result = es.search(body={"query": {"match_all": {}}, "size":max_size, "_source":False, "fields": ['properties.osm_id']}, index = index_name)
    osmids_el = [e['fields']['properties.osm_id'][0] for e in result['hits']['hits']]
    return osmids_el

def get_keyword_frequency(es, index_name: str, keyword: str, size = 100000):
    keyword_frequency = search_by_keyword_elastic(es, index_name, keyword, size = size, count = True)
    return keyword_frequency

def get_keywords_frequencies(es, index_name: str, keywords: list, size = 100000):
    keywords_frequencies = {}
    for keyword in keywords:
        keywords_frequencies[keyword] = get_keyword_frequency(es, index_name, keyword, size = size)
    return keywords_frequencies



def search_by_keyword_elastic(es, index_name, kw:str, size = 100000, count = False):
    query = {
        "query": {
            "bool":{
                "filter": {
                    "match": {
                        "properties.keywords.keyword": kw
                    }
                }
            }
        }
    }
    if count:
        return es.count(index = index_name, body = query)['count']
    else:
        query["fields"] = ["geometry", "properties.osm_id"]
        query["_source"] = False
        query["size"] = size
        return query


# search by keyword, and (min,max) distance to a given point, specifying or not the disjoint requirement with relation to a given geo_shape
def search_by_keyword_and_distance_elastic(es, index_name, kw: str, lij: float, uij: float, center, require_disjoint = False,
                                   base_hit = None, size = 100000, count = False, ordered = False):
    lon, lat = center
    if lij == 0:
        query = {
            "query": {
                "bool": {
                    "filter": [
                        {"match": {"properties.keywords.keyword": kw}},
                        {
                            "geo_distance": {
                                "distance": f"{uij}m",
                                "centroid": {
                                    "lat": lat,
                                    "lon": lon
                                }
                            }
                        }
                    ]
                }
            }
        }
    else:
        query = {
            "query": {
                "bool": {
                    "must_not": {
                        "geo_distance": {
                            "distance": f"{lij}m",
                            "centroid": [lon, lat]
                        }
                    },
                    "filter": [
                        {"match": {"properties.keywords.keyword": kw}},
                        {
                            "geo_distance": {
                                "distance": f"{uij}m",
                                "centroid": {
                                    "lat": lat,
                                    "lon": lon
                                }
                            }
                        }
                    ]
                }
            }
        }
    if require_disjoint:
        query["query"]["bool"]["filter"].append({
            "geo_shape": {
              "geometry": {
                "indexed_shape": {
                  "index": index_name,
                  "id": base_hit['_id'],
                  "path": "geometry"
                },
                "relation": 'disjoint' # intersects, contains, within
              }
            }
        })
    if count:
        return es.count(index = index_name, body = query)['count']
    else:
        query["fields"] = ["centroid", "geometry", "properties.osm_id"]
        query["_source"] = False
        query["size"] = size
        if ordered is True:
            query["sort"] =  [
                {
                    "_geo_distance" : {
                        "centroid" : [lon, lat],
                        "order" : "asc",
                        "unit" : "m",
                        "mode" : "min",
                        "distance_type" : "arc",
                        "ignore_unmapped": True
                    }
                }
            ]
        return query
    

def search_by_keyword_and_relation_elastic(es, index_name, kw:str, hit: dict, relation, size = 100000, count = False):
    query = {
        "query": {
            "bool": {
                "filter": [
                    {"match": {"properties.keywords.keyword": kw}},
                    {
                        "geo_shape": {
                        "geometry": {
                            "indexed_shape": {
                            "index": index_name,
                            "id": hit['_id'],
                            "path": "geometry"
                            },
                            "relation": relation # intersects, contains, within
                        }
                        }
                    }
                ]
            }
        }
    }
    if count:
        return es.count(index = index_name, body = query)['count']
    else:
        query["fields"] = ["geometry", "properties.osm_id"]
        query["_source"] = False
        query["size"] = size
        return query
    
def find_qq_e_matches_intersecting_elastic(es, index_name, wi: str, wj: str, relation: str, candidate_objects = {}, changed = False, size = 100000):
    # candidate_objects must be a dictionary whose keys are string keywords. Example of overall structure:
    
    #geom1 = search_by_osm_id(es, -13383295)['hits']['hits'][0]['_source']['geometry']
    #candidate_objects = {'bank': [
    #    {'fields': {'properties.osm_id': [-13383295], 'geometry': [geom1]}}, 
    #    {'fields': {'properties.osm_id': [591023793], 'geometry': [geom2]}}
    #]}
    
    if relation not in ['within', 'contains', 'intersects']:
        print('Invalid relation name. Use one among "within", "contains" and "intersects"')
        return None
        
    invert_relation = {
        'within': 'contains',
        'contains': 'within',
        'intersects': 'intersects'
    }

    if wi in candidate_objects and wj in candidate_objects:
        if len(candidate_objects[wi]) > len(candidate_objects[wj]):
            wi, wj = wj, wi
            relation = invert_relation[relation]
            changed = True
        result = candidate_objects[wi]

    elif wi in candidate_objects:
        result = candidate_objects[wi]

    elif wj in candidate_objects:
        wi, wj = wj, wi
        relation = invert_relation[relation]
        changed = True
        result = candidate_objects[wi]

    else:
        total_objs_wi = search_by_keyword_elastic(es, index_name, wi, count = True)
        total_objs_wj = search_by_keyword_elastic(es, index_name, wj, count = True)
        if total_objs_wi > total_objs_wj:
            wi, wj = wj, wi
            relation = invert_relation[relation]
            changed = True
        result_query = search_by_keyword_elastic(es, index_name, wi, size = size)
        result = es.search(index = index_name, body = result_query)['hits']['hits']

    qq_e_matches = []
    if wj in candidate_objects:
        ids_candidate_wj = [e['fields']['properties.osm_id'] for e in candidate_objects[wj]]
    
    request = []
    for hit_i in result:
        req_head = {'index': index_name}
        req_body = search_by_keyword_and_relation_elastic(es, index_name, wj, hit_i, invert_relation[relation], size = size)
        request.extend([req_head, req_body])
        
    responses = es.msearch(body = request)['responses']
    for i, hit_i in enumerate(result):
        #try:
        result2 = responses[i]['hits']['hits']
        # except:
        #     print('i:', i)
        #     print('responses[i]:', responses[i])
        if wj in candidate_objects:
            result2 = [e for e in result2 if e['fields']['properties.osm_id'] in ids_candidate_wj]
        for hit_j in result2:
            if changed:
                qq_e_matches.append((hashabledict(**hit_j), hashabledict(**hit_i)))
            else:
                qq_e_matches.append((hashabledict(**hit_i), hashabledict(**hit_j)))
    return qq_e_matches
    
    
def find_qq_e_matches_inclusive_norelation_elastic(es, index_name, wi: str, wj: str, lij: float, uij: float, 
                                                   require_disjoint = False, candidate_objects = {}, changed = False):
    
    if wi in candidate_objects and wj in candidate_objects:
        if len(candidate_objects[wi]) > len(candidate_objects[wj]):
            wi, wj = wj, wi
            changed = True
        result = candidate_objects[wi]

    elif wi in candidate_objects:
        result = candidate_objects[wi]

    elif wj in candidate_objects:
        wi, wj = wj, wi
        changed = True
        result = candidate_objects[wi]

    else:
        total_objs_wi = search_by_keyword_elastic(es, index_name, wi, count = True)
        total_objs_wj = search_by_keyword_elastic(es, index_name, wj, count = True)
        if total_objs_wi > total_objs_wj:
            wi, wj = wj, wi
            changed = True
        result_query = search_by_keyword_elastic(es, index_name, wi)
        result = es.search(index = index_name, body = result_query)['hits']['hits']

    qq_e_matches = []
    if wj in candidate_objects:
        ids_candidate_wj = [e['fields']['properties.osm_id'] for e in candidate_objects[wj]]

    request = []
    for hit_i in result:
        lng, lat = shape(hit_i['fields']['geometry'][0]).centroid.coords[0]
        req_head = {'index': index_name}
        req_body = search_by_keyword_and_distance_elastic(es, index_name, wj, lij, uij, (lng, lat), require_disjoint, hit_i)
        request.extend([req_head, req_body])
    responses = es.msearch(body = request)['responses']
    
    for i, hit_i in enumerate(result):
        result2 = responses[i]['hits']['hits']
        if wj in candidate_objects:
            result2 = [e for e in result2 if e['fields']['properties.osm_id'] in ids_candidate_wj]
        for hit_j in result2:
            if changed:
                qq_e_matches.append((hashabledict(**hit_j), hashabledict(**hit_i)))
            else:
                qq_e_matches.append((hashabledict(**hit_i), hashabledict(**hit_j)))
    return qq_e_matches

def find_qq_e_matches_one_way_exclusion_norelation_elastic(es, index_name, wi: str, wj: str, lij: float, uij: float, require_disjoint = False, 
                                                   candidate_objects = {}, invert_results = False):
    if lij == 0:
        return find_qq_e_matches_inclusive_norelation_elastic(es, index_name, wi, wj, lij, uij, require_disjoint, candidate_objects)

    if wi in candidate_objects:
        result = list(candidate_objects[wi])

    else:
        result_query = search_by_keyword_elastic(es, index_name, wi)
        result = es.search(index = index_name, body = result_query)['hits']['hits']

    qq_e_matches = []
    
    if wj in candidate_objects:
        ids_candidate_wj = [e['fields']['properties.osm_id'] for e in candidate_objects[wj]]
            
    request = []
    for hit_i in result:
        lng, lat = shape(hit_i['fields']['geometry'][0]).centroid.coords[0]
        req_head = {'index': index_name}
        req_body = search_by_keyword_and_distance_elastic(es, index_name, wj, 0, lij, (lng, lat), count = False, size = 1)
        request.extend([req_head, req_body])
    responses = es.msearch(body = request)['responses']

    for i, hit_i in enumerate(result):
        total_near = responses[i]['hits']['total']['value']
        if total_near > 0:
            result[i] = None
    result = [e for e in result if e is not None]
    if len(result) == 0:
        return []

    request = []
    for hit_i in result:
        lng, lat = shape(hit_i['fields']['geometry'][0]).centroid.coords[0]
        req_head = {'index': index_name}
        req_body = search_by_keyword_and_distance_elastic(es, index_name, wj, lij, uij, (lng, lat), require_disjoint, hit_i)
        request.extend([req_head, req_body])
    responses = es.msearch(body = request)['responses']

    for i, hit_i in enumerate(result):
        result2 = responses[i]['hits']['hits']
        if wj in candidate_objects:
            result2 = [e for e in result2 if e['fields']['properties.osm_id'] in ids_candidate_wj]
        for hit_j in result2:
            if invert_results:
                qq_e_matches.append((hashabledict(**hit_j), hashabledict(**hit_i)))
            else:
                qq_e_matches.append((hashabledict(**hit_i), hashabledict(**hit_j)))
            
    return qq_e_matches

def find_qq_e_matches_mutual_exclusion_norelation_elastic(es, index_name, wi: str, wj: str, lij: float, uij: float, require_disjoint = False, 
                                                  candidate_objects = {}, debug = False):
    if lij == 0:
        return find_qq_e_matches_inclusive_norelation_elastic(es, index_name, wi, wj, lij, uij, require_disjoint, candidate_objects)


    # if wi in candidate_objects:
    #     result = candidate_objects[wi]

    # else:
    #     result_query = search_by_keyword_elastic(es, index_name, wi)
    #     result = es.search(index = index_name, body = result_query)['hits']['hits']
    
    # if wj in candidate_objects:
    #     ids_candidate_wj = [e['fields']['properties.osm_id'] for e in candidate_objects[wj]]

    qq_e_matches_part1 = find_qq_e_matches_one_way_exclusion_norelation_elastic(es, index_name, wi, wj, lij, uij, require_disjoint, 
                                                   candidate_objects, invert_results = False)

    for vertex in candidate_objects:
        if len(candidate_objects[vertex]) == 0:
            if debug:
                print(f'Zero candidate objects for vertex {vertex}')
            return []

    qq_e_matches_part2 = find_qq_e_matches_one_way_exclusion_norelation_elastic(es, index_name, wj, wi, lij, uij, require_disjoint, 
                                                   candidate_objects, invert_results = True)

    ids_pairs_part1 = [(s[0]['fields']['properties.osm_id'][0], s[1]['fields']['properties.osm_id'][0]) for s in qq_e_matches_part1]
    qq_e_matches = list(filter(lambda s: (s[0]['fields']['properties.osm_id'][0], s[1]['fields']['properties.osm_id'][0]) in ids_pairs_part1
           , qq_e_matches_part2))
    #qq_e_matches = list(set(qq_e_matches_part1).intersection(set(qq_e_matches_part2)))
    return qq_e_matches

#finding the e-matches for the edge 
def find_qq_e_matches_elastic(es, index_name, edge, candidate_objects = None, debug = False):

    wi = edge.vi.keyword
    wj = edge.vj.keyword
    lij = edge.constraint['lij']
    uij = edge.constraint['uij']
    sign = edge.constraint['sign']
    relation = edge.constraint['relation']
    if candidate_objects is None:
        candidate_objects = {}
    
    if relation in ['contains', 'within', 'intersects']:
        return find_qq_e_matches_intersecting_elastic(es, index_name, wi, wj, relation, candidate_objects = candidate_objects)

    elif relation == 'disjoint':
        require_disjoint = True

    elif relation is None:
        require_disjoint = False

    else:
        print('Invalid relation! Use None, or one among "intersects", "contains", "within", "disjoint"')
        return
        
    if sign == '-':
        return find_qq_e_matches_inclusive_norelation_elastic(es, index_name, wi, wj, lij, uij, require_disjoint, 
                                                              candidate_objects = candidate_objects)
    elif sign == '>':
        return find_qq_e_matches_one_way_exclusion_norelation_elastic(es, index_name, wi, wj, lij, uij, require_disjoint, 
                                                                      candidate_objects = candidate_objects)
    elif sign == '<':
        return find_qq_e_matches_one_way_exclusion_norelation_elastic(es, index_name, wj, wi, lij, uij, require_disjoint, 
                                                                      candidate_objects = candidate_objects, invert_results = True)
    elif sign == '<>':
        return find_qq_e_matches_mutual_exclusion_norelation_elastic(es, index_name, wi, wj, lij, uij, require_disjoint, 
                                                                     candidate_objects = candidate_objects, debug = debug)
    else:
        print('Invalid edge sign! Use one among: "-", ">", "<", "<>"')
        return None

def compute_qq_e_matches_for_all_edges_greedy_elastic(es, index_name, sp, alternated=False, debug = False):
    sp_keywords = [v.keyword for v in sp.vertices]
    keyword_frequencies = get_keywords_frequencies(es, index_name, sp_keywords)
    qq_e_matches = {}
    if debug:
        print(f'Keyword frequencies: {keyword_frequencies}')
    for kw in keyword_frequencies:
        if keyword_frequencies[kw] == 0:
            if debug:
                print(f'Zero solutions, since keyword {kw} is not found in dataset')
            return
    reverse=False
    initial_vertex = sorted(sp.vertices, key = lambda v: keyword_frequencies[v.keyword], reverse=reverse)[0]
    if alternated:
        reverse = not reverse
    remaining_edges = set([(e.vi, e.vj) for e in sp.edges])
    processed_edges = []
    current_vertex = initial_vertex
    vertices_sequence = [current_vertex]
    visited_vertices_with_remaining_edges = set()#{current_vertex}
    if debug:
        print(f'Initial vertex: {current_vertex.keyword}')

    candidate_objects = {}
    candidates_initial_vertex_query = search_by_keyword_elastic(es, index_name, current_vertex.keyword)
    candidates_initial_vertex = es.search(index = index_name, body = candidates_initial_vertex_query)['hits']['hits']
    if debug:
        print(f'Total candidates for initial vertex: {len(candidates_initial_vertex)}')
    for i, candidate in enumerate(candidates_initial_vertex):
        candidates_initial_vertex[i] = hashabledict(**candidate)
    candidate_objects[current_vertex.keyword] = set(candidates_initial_vertex)
    while len(remaining_edges) > 0:
        for vertex in candidate_objects:
            if len(candidate_objects[vertex]) == 0:
                print(f'Zero candidate objects for vertex {vertex}')
                return None
        
        candidates_next_vertex = set(sp.neighbors[current_vertex])
        for e in processed_edges:
            if e[0] == current_vertex:
                candidates_next_vertex -= {e[1]}
            if e[1] == current_vertex:
                candidates_next_vertex -= {e[0]}
                
        if len(candidates_next_vertex) > 0:
            visited_vertices_with_remaining_edges.add(current_vertex)
            next_vertex = sorted(candidates_next_vertex, key = lambda v: keyword_frequencies[v.keyword], reverse=reverse)[0]
            if alternated:
                reverse = not reverse
            if debug:
                print(f'Next vertex: {next_vertex.keyword}')
            edge = sp.pairs_to_edges[current_vertex][next_vertex]
            if edge not in qq_e_matches:
                if debug:
                    print(f'Computing qq-e-matches for edge {edge.id}')
                qq_e_matches[edge] = find_qq_e_matches_elastic(es, index_name, edge, candidate_objects = candidate_objects, debug = debug)
                if debug:
                    print(f'Total qq-e-matches for edge {edge.id}: {len(qq_e_matches[edge])}')
                if len(qq_e_matches[edge]) == 0:
                    return None
                update_candidate_objects(candidate_objects, edge, qq_e_matches[edge])
                keyword_frequencies[edge.vi.keyword] = min(keyword_frequencies[edge.vi.keyword], len(candidate_objects[edge.vi.keyword]))
                keyword_frequencies[edge.vj.keyword] = min(keyword_frequencies[edge.vj.keyword], len(candidate_objects[edge.vj.keyword]))
                processed_edges.append((current_vertex, next_vertex))
                remaining_edges -= {(current_vertex, next_vertex), (next_vertex, current_vertex)}
            current_vertex = next_vertex
            vertices_sequence.append(current_vertex)
            

        else:
            visited_vertices_with_remaining_edges -= {current_vertex}
            if len(visited_vertices_with_remaining_edges) == 0:
                break
            current_vertex = sorted(visited_vertices_with_remaining_edges, key = lambda v: keyword_frequencies[v.keyword], reverse=reverse)[0]
            if alternated:
                reverse = not reverse
            vertices_sequence.append(current_vertex)
            if debug:
                print(f'Next vertex: {current_vertex.keyword}')
    if debug:
        print(f'Processed vertices sequence: {vertices_sequence}')
        print(f'Processed edges sequence: {processed_edges}')
    return qq_e_matches


def generate_partial_solution_from_qq_e_match(qq_e_match, edge, sp):
    os, ot = qq_e_match
    partial_solution = {vertex.id: None for vertex in sp.vertices}
    partial_solution[edge.vi.id] = os
    partial_solution[edge.vj.id] = ot
    return partial_solution

def generate_partial_solutions_from_qq_e_matches(qq_e_matches, edge, sp, pool_obj):
    generate_partial_solution_from_qq_e_match_partial = partial(generate_partial_solution_from_qq_e_match, edge = edge, sp = sp)
    if pool_obj is not None: #parallel
        partial_solutions = pool_obj.map(generate_partial_solution_from_qq_e_match_partial, qq_e_matches)
    else: #sequential
        partial_solutions = [generate_partial_solution_from_qq_e_match_partial(qqem) for qqem in qq_e_matches]
    return partial_solutions


def merge_partial_solutions(pa, pb, sp):
    # pa and pb and dictionaries in the format: {v1: obj1, ..., vn: objn} where vi's are vertices and obji's are GeoObjs
    # merging means aggregating the two partial solutions into a single one if possible
    # sometimes it's not possible, when the two solutions provide a different value for the same vertex
    merged = dict()
    for vertex in sp.vertices:
        if pa[vertex.id] is not None and pb[vertex.id] is not None and not hits_are_equal(pa[vertex.id], pb[vertex.id]):
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


def filter_qq_e_matches_by_vertex_candidates_elastic(qq_e_matches, edge, candidates):
    ids_candidate_vi = [c['fields']['properties.osm_id'][0] for c in candidates[edge.vi]]
    ids_candidate_vj = [c['fields']['properties.osm_id'][0] for c in candidates[edge.vj]]
    return list(filter(lambda qqem: (qqem[0]['fields']['properties.osm_id'][0] in ids_candidate_vi and \
                                     qqem[1]['fields']['properties.osm_id'][0] in ids_candidate_vj), qq_e_matches))
    
def join_qq_e_matches_elastic(sp, qq_e_matches: dict, alternated, debug = False, pool_obj = None):
    edges = sp.edges
    vertices = sp.vertices
    candidates = {vertex: set() for vertex in vertices}
    for edge in edges:
        cvi = [x[0] for x in qq_e_matches[edge]]
        cvj = [x[1] for x in qq_e_matches[edge]]
        if candidates[edge.vi] == set(): 
            candidates[edge.vi] = set(cvi)
        else: 
            ids_cvi = [c['fields']['properties.osm_id'][0] for c in cvi]
            candidates[edge.vi] = set(filter(lambda c: c['fields']['properties.osm_id'][0] in ids_cvi, candidates[edge.vi]))
        if candidates[edge.vj] == set(): 
            candidates[edge.vj] = set(cvj)
        else: 
            ids_cvj = [c['fields']['properties.osm_id'][0] for c in cvj]
            candidates[edge.vj] = set(filter(lambda c: c['fields']['properties.osm_id'][0] in ids_cvj, candidates[edge.vj]))
    
        
    for edge in edges:
        qq_e_matches[edge] = filter_qq_e_matches_by_vertex_candidates_elastic(qq_e_matches[edge], edge, candidates)
        if debug and  len(qq_e_matches[edge]) == 0:
            print(f'Zero qq-e-matches for edge {edge.id} after candidates filtering for joining')
    edges.sort(key = lambda edge: len(qq_e_matches[edge]))
    if len(edges) >= 3 and alternated:
        edges_values_dict = {e: len(qq_e_matches[e]) for e in edges}
        edges = get_edges_order(edges_values_dict, edges, sp, alternated = alternated, debug=debug)

    partial_solutions = [{vertex.id: None for vertex in sp.vertices}]
    for edge in edges:
        partial_solutions_edge = generate_partial_solutions_from_qq_e_matches(qq_e_matches[edge], edge, sp, pool_obj)
        partial_solutions = merge_lists_of_partial_solutions(partial_solutions, partial_solutions_edge, sp)
        
    final_solutions = list(filter(lambda x: x is not None, partial_solutions))
    return final_solutions

def get_edges_of_vertex(vertex, sp):
    return list(sp.pairs_to_edges[vertex].values())

def get_edges_order(edges_values_dict, initial_edges_order, sp, alternated: bool, debug = False):
    if debug:
        print(f'Edges and values dict: {edges_values_dict}')
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

def QQSPM_ELASTIC(sp, index_name, es_instance = None, config_dir = "config", alternated=True, parallel = False, debug = False):
    global es
    global current_index_name

    if es_instance is not None: 
        es = es_instance
    elif current_index_name != index_name:
        current_index_name = index_name
        if es is not None:
            es.close()
        es = establish_elastic_connection(config_dir = config_dir, debug = debug)
        
    t0 = time()
    if parallel:
        pool_obj = ThreadPool(int(multiprocessing.cpu_count()-3)) #ThreadPool
    else:
        pool_obj = None
    qq_e_matches = compute_qq_e_matches_for_all_edges_greedy_elastic(es, index_name, sp, debug = debug)
    if qq_e_matches is None:
        solutions = []
    else:
        solutions = join_qq_e_matches_elastic(sp, qq_e_matches, alternated=alternated, debug = debug, pool_obj=pool_obj)
    memory_usage = resource_monitoring.get_total_memory_usage_for_elastic()
    if pool_obj is not None:
        pool_obj.close()
    elapsed_time = time() - t0
    return solutions, elapsed_time, memory_usage
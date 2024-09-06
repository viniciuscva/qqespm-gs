from qqespm_quadtree_CGA import SpatialEdge, SpatialVertex, SpatialPatternGraph
import re


relations_to_postgis_functions = {
    'equals': 'ST_Equals', 
    'touches': 'ST_Touches', 
    'overlaps': 'ST_Overlaps', 
    'covers': 'ST_Contains', 
    'contains': 'ST_Contains',
    'coveredby': 'ST_Within', 
    'within': 'ST_Within',
    'disjoint': 'ST_Disjoint',
    'intersects': 'ST_Intersects'
}

def distance_constraint(keyword1: str, keyword2: str, min_distance: float, max_distance: float,
                        first_excludes_second: bool, second_excludes_first: bool) -> dict:
    if min_distance<0 or max_distance<=min_distance:
        print('Invalid min or max distance')
        return None
    if keyword1 == keyword2:
        print('Invalid pair of keywords. They must be different from each other')
        return None
    if first_excludes_second and second_excludes_first:
        exclusion_sign = '<>'
    elif first_excludes_second:
        exclusion_sign = '>'
    elif second_excludes_first:
        exclusion_sign = '<'
    else:
        exclusion_sign = '-'
    return {
        'keyword1': keyword1,
        'keyword2': keyword2,
        'min_distance': min_distance,
        'max_distance': max_distance,
        'exclusion_sign': exclusion_sign
    }

def connectivity_constraint(keyword1: str, keyword2: str, topological_relation: str):
    if not topological_relation in ['intersects', 'contains', 'within']:
        print('Invalid topological relation. Choose one among (intersects,contains,within)')
        return None
    return {
        'keyword1': keyword1,
        'keyword2': keyword2,
        'topological_relation': topological_relation,
    }

def spatial_pattern(constraints: list[dict]):
    vertices = []
    edges = []
    for constraint in constraints:
        wi, wj = constraint['keyword1'], constraint['keyword2']
        added_keywords = [v.keyword for v in vertices]
        if wi in added_keywords:
            vi = vertices[added_keywords.index(wi)]
        else:
            vi = SpatialVertex(new_id(), wi)
            vertices.append(vi)
        if wj in added_keywords:
            vj = vertices[added_keywords.index(wj)]
        else:
            vj = SpatialVertex(new_id(), wj)
            vertices.append(vj)
        added_edges = [(e.vi.keyword, e.vj.keyword) for e in edges]
        if (wi,wj) in added_edges or (wj,wi) in added_edges:
            if (wi,wj) in added_edges:
                existing_edge = edges[added_edges.index((wi,wj))]
            else:
                existing_edge = edges[added_edges.index((wj,wi))]
            if 'topological_relation' in constraint:
                existing_edge.constraint['relation'] = constraint['topological_relation']
            if 'min_distance' in constraint:
                existing_edge.constraint['lij'] = constraint['min_distance']
            if 'max_distance' in constraint:
                existing_edge.constraint['uij'] = constraint['max_distance']
            if 'exclusion_sign' in constraint:
                existing_edge.constraint['sign'] = constraint['exclusion_sign']
        else:
            relation = constraint.get('topological_relation')
            lij = constraint.get('min_distance') or 0
            uij = constraint.get('max_distance') or 10000
            sign = constraint.get('exclusion_sign') or '-'
            edge = SpatialEdge(f"{vi.id}-{vj.id}", vi, vj, lij, uij, sign, relation)
            edges.append(edge)
    if len(vertices) < 2 or len(edges) == 0:
        print('Did not provide enough vertices or edges to create spatial pattern.')
        return None
    spatial_pattern_json = SpatialPatternGraph(vertices, edges).to_json()
    return spatial_pattern_json




current_id = -1

def new_id():
    global current_id
    current_id = current_id + 1
    return current_id


def convert_template_expression_into_sp(expression):
    vertices = []
    edges = []
    try:
        requirements_expressions = expression.split('&')
        for requirement_exp in requirements_expressions:
            between_expression_match = re.match(r"([a-zA-Z]{0,50}) between (\d+) and (\d+) meters from ([a-zA-Z]{0,50})", requirement_exp)
            if between_expression_match:
                wi, lij, uij, wj = between_expression_match.groups()
                lij = int(lij)
                uij = int(uij)
                current_keywords = [v.keyword for v in vertices]
                if wi in current_keywords:
                    vi = vertices[current_keywords.index(wi)]
                else:
                    vi = SpatialVertex(new_id(), wi)
                    vertices.append(vi)
                if wj in current_keywords:
                    vj = vertices[current_keywords.index(wj)]
                else:
                    vj = SpatialVertex(new_id(), wj)
                    vertices.append(vj)
                edge = SpatialEdge(f"{vi.id}-{vj.id}", vi, vj, lij, uij)
                edges.append(edge)
                continue
            within_dist_expression_match = re.match(r"([a-zA-Z]{0,50}) within (\d+) meters from ([a-zA-Z]{0,50})", requirement_exp)
            if within_dist_expression_match:
                wi, uij, wj = between_expression_match.groups()
                uij = int(uij)
                current_keywords = [v.keyword for v in vertices]
                if wi in current_keywords:
                    vi = vertices[current_keywords.index(wi)]
                else:
                    vi = SpatialVertex(new_id(), wi)
                    vertices.append(vi)
                if wj in current_keywords:
                    vj = vertices[current_keywords.index(wj)]
                else:
                    vj = SpatialVertex(new_id(), wj)
                    vertices.append(vj)
                edge = SpatialEdge(f"{vi.id}-{vj.id}", vi, vj, 0, uij)
                edges.append(edge)
                continue
            connectivity_expression_match = re.match(r"([a-zA-Z]{0,50}) (intersects|contains|within) ([a-zA-Z]{0,50})", requirement_exp)
            if connectivity_expression_match:
                wi, relation, wj = between_expression_match.groups()
                current_keywords = [v.keyword for v in vertices]
                if wi in current_keywords:
                    vi = vertices[current_keywords.index(wi)]
                else:
                    vi = SpatialVertex(new_id(), wi)
                    vertices.append(vi)
                if wj in current_keywords:
                    vj = vertices[current_keywords.index(wj)]
                else:
                    vj = SpatialVertex(new_id(), wj)
                    vertices.append(vj)
                edge = SpatialEdge(f"{vi.id}-{vj.id}", vi, vj, 0, float('inf'), relation=relation)
                edges.append(edge)
                continue
            # if the program reaches this line, that's because the parsing failed
            return
        sp = SpatialPatternGraph(vertices, edges)
        return sp

    except:
        return None
    

def get_greedy_search_path_by_keywords_frequencies(sp, keyword_frequencies, alternated, debug = False):
    if debug:
        print(f'Keyword frequencies: {keyword_frequencies}')
    reverse = False # if true, calculates the max, if false, calculates the min
    initial_vertex = sorted(sp.vertices, key = lambda v: keyword_frequencies[v.keyword], reverse=reverse)[0]
    if alternated:
        reverse = not reverse
    remaining_edges = set([(e.vi, e.vj) for e in sp.edges])
    processed_edges = []
    current_vertex = initial_vertex
    vertices_sequence = [current_vertex]
    visited_vertices_with_remaining_edges = set()
    while len(remaining_edges) > 0:
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
            processed_edges.append((current_vertex, next_vertex))
            remaining_edges -= {(current_vertex, next_vertex), (next_vertex, current_vertex)}
            current_vertex = next_vertex
            if current_vertex not in vertices_sequence:
                vertices_sequence.append(current_vertex)
        else:
            visited_vertices_with_remaining_edges -= {current_vertex}
            if len(visited_vertices_with_remaining_edges) == 0:
                return vertices_sequence
            current_vertex = sorted(visited_vertices_with_remaining_edges, key = lambda v: keyword_frequencies[v.keyword], reverse=reverse)[0]
            if alternated:
                reverse = not reverse
            if current_vertex not in vertices_sequence:
                vertices_sequence.append(current_vertex)

    return vertices_sequence

def condition_for_keyword(keyword_column_names:list, keyword:str, boolean_connector:str = 'OR'):
    #boolean_connector must be either 'AND' or 'OR'
    expression = f"{keyword_column_names[0]} = '{keyword}' "
    for column in keyword_column_names[1:]:
        expression += f"{boolean_connector} {column} = '{keyword}' "
    return expression



def build_exclusion_check(lij, sign, tb_vi_name, tb_vj_name, approximation_buffer = 7):
    exclusion_check = ''
    if sign == '>':
        exclusion_check += f"""NOT EXISTS (SELECT 1 FROM {tb_vj_name} aux WHERE ST_DWithin(aux.centroid::geography, {tb_vi_name}.centroid::geography, {lij}, false)) """
    elif sign == '<':
        exclusion_check += f"""NOT EXISTS (SELECT 1 FROM {tb_vi_name} aux WHERE ST_DWithin({tb_vj_name}.centroid::geography, aux.centroid::geography, {lij}, false)) """
    elif sign == '<>':
        exclusion_check += f"""NOT EXISTS (SELECT 1 FROM {tb_vj_name} aux1 WHERE ST_DWithin(aux1.centroid::geography, {tb_vi_name}.centroid::geography, {lij}, false)) AND\n"""
        exclusion_check += f"""NOT EXISTS (SELECT 1 FROM {tb_vi_name} aux2 WHERE ST_DWithin({tb_vj_name}.centroid::geography, aux2.centroid::geography, {lij}, false))"""
    return exclusion_check

def with_clause_temporary_table_keyword(keyword, pois_table_name, keyword_column_names = ['amenity','shop','tourism']):
    return f"""    _tb_{keyword} AS
    (SELECT * FROM {pois_table_name} WHERE {condition_for_keyword(keyword_column_names, keyword)})"""

def with_clause_temporary_tables_all_keywords(keywords, keyword_column_names = ['amenity','shop','tourism']):
    return 'WITH\n' + ',\n'.join([with_clause_temporary_table_keyword(keyword, keyword_column_names = keyword_column_names) for keyword in keywords])

def select_clause_all_keywords(keywords, use_alias = True, include_centroids = False):
    temporary_table_names = [f'_tb_{k}' for k in keywords]

    if use_alias:
        select_clause = 'SELECT ' + ', '.join([f'{ttn}.osm_id AS {ttn}_id' for ttn in temporary_table_names])
    else:
        select_clause = 'SELECT ' + ', '.join([f'{ttn}_id' for ttn in temporary_table_names])
    if include_centroids:
        if use_alias:
            select_clause += ',\n' + ', '.join([f'{ttn}.centroid AS {ttn}_centroid' for ttn in temporary_table_names])
        else:
            select_clause += ',\n' + ', '.join([f'{ttn}_centroid' for ttn in temporary_table_names])
    return select_clause

def from_clause_all_keywords(keywords):
    temporary_table_names = [f'_tb_{k}' for k in keywords]
    return 'FROM ' + ', '.join(temporary_table_names)


def condition_clause_for_edge(edge):
    wi = edge.vi.keyword
    wj = edge.vj.keyword
    tb_vi_name = f'_tb_{wi}'
    tb_vj_name = f'_tb_{wj}'
    lij, uij = edge.constraint['lij'], edge.constraint['uij']
    sign = edge.constraint['sign']
    relation = edge.constraint['relation']
    distance_check = ''
    if lij > 0 and uij < float('inf'):
        distance_check += f'ST_DistanceSphere({tb_vi_name}.centroid, {tb_vj_name}.centroid) BETWEEN {lij} AND {uij} '
    elif lij == 0 and uij < float('inf'):
        distance_check += f'ST_DistanceSphere({tb_vi_name}.centroid, {tb_vj_name}.centroid) <= {uij} '
    elif lij > 0 and uij == float('inf'):
        distance_check += f'ST_DistanceSphere({tb_vi_name}.centroid, {tb_vj_name}.centroid) >= {lij} '
    if edge.constraint['is_exclusive']:
        exclusion_check = build_exclusion_check(lij, sign, tb_vi_name, tb_vj_name)
    else:
        exclusion_check = ''
    if relation is not None:
        relation_check = relations_to_postgis_functions[relation] + f'({tb_vi_name}.geometry, {tb_vj_name}.geometry)'
    else:
        relation_check = ''
    return ' AND \n'.join([check for check in [distance_check, exclusion_check, relation_check] if check != ''])

def condition_clause_for_multiple_edges(edges):
    return '    ' + (' AND \n'.join([condition_clause_for_edge(e) for e in edges])).replace("\n", "\n    ")


def edges_of_vertex_and_previous(vertex, previous_vertices, sp):
    """
    Given a vertex, and a list of previously visited vertices,
    returns all edges in the spatial pattern sp which connect the given vertex with any of the vertices in the previous_vertices list.
    """
    edges_to_be_pŕocessed = []
    for prev_vertex in previous_vertices:
        potential_edge = sp.pairs_to_edges.get(vertex).get(prev_vertex)
        if potential_edge is not None:
            edges_to_be_pŕocessed.append(potential_edge)
    return edges_to_be_pŕocessed
    

def build_sql_query_for_spatial_pattern(sp_json, vertices_sequence_greedy, pois_table_name, keyword_column_names = ['amenity','shop','tourism'], limit = 0):
    sp = SpatialPatternGraph.from_json(sp_json)    
    if limit > 0:
        limit_check = f'LIMIT {limit}'
    else:
        limit_check = ''
    keywords_sequence_greedy = [v.keyword for v in vertices_sequence_greedy]
    keywords_sequence_in_sp = [v.keyword for v in sp.vertices]

    sql = f"""{with_clause_temporary_tables_all_keywords(keywords_sequence_greedy, pois_table_name, keyword_column_names = keyword_column_names)}
    \r{select_clause_all_keywords(keywords_sequence_in_sp)}
    \rFROM _tb_{keywords_sequence_greedy[0]}
    """
    for i, vertex in enumerate(vertices_sequence_greedy[1:]):
        keyword = vertex.keyword
        sql += f"""\rINNER JOIN _tb_{keyword}
        \rON
        \r{condition_clause_for_multiple_edges(edges_of_vertex_and_previous(vertex, vertices_sequence_greedy[:i+1], sp))}
        """
    sql += f"""{limit_check}"""

    return sql    

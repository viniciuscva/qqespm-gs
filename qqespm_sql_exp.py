#import __init__
import psycopg2
from geopandas import GeoDataFrame
from time import time
import psutil
from configparser import ConfigParser
from itertools import chain
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# this module creates postgis queries with explicit joins following alternated strategy by total frequency of the keywords

relations_to_postgis_functions = {
    'equals': 'ST_Equals', 
    'touches': 'ST_Touches', 
    'overlaps': 'ST_Overlaps', 
    'covers': 'ST_Covers', 
    'contains': 'ST_Covers',
    'coveredby': 'ST_CoveredBy', 
    'within': 'ST_CoveredBy',
    'disjoint': 'ST_Disjoint',
    'intersects': 'ST_Intersects'
}

DEFAULT_CONFIG_FILENAME = 'config/london_pois_bbox_100perc.ini'
current_config_filename = None
conn = None
COLUMN_NAMES = None

def get_keyword_frequency(conn, keyword:str, column_names, debug = False):
    sql = f"""
    SELECT count(*)
    FROM pois
    WHERE ({condition_for_keyword(column_names, keyword)})
    """
    if debug:
        print(f'SQL query to be executed: {sql}')
    cur = conn.cursor()
    cur.execute(sql)
    frequency = cur.fetchall()[0][0]
    cur.close()
    return frequency

def get_keywords_frequencies(conn, keywords: list, column_names = ['amenity','shop','tourism'], debug = False):
    keywords_frequencies = {}
    for keyword in keywords:
        keywords_frequencies[keyword] = get_keyword_frequency(conn, keyword, column_names = column_names, debug = debug)
    return keywords_frequencies

def get_keyword_columns(conn):
    cursor = conn.cursor()
    cursor.execute("select column_name from information_schema.columns where table_schema = 'public' and table_name = 'pois';")
    result = cursor.fetchall()
    column_names = list(set(chain(*result)) - {'osm_id', 'geometry', 'name', 'centroid', 'lon', 'lat', 'id'})
    cursor.close()
    return sorted(column_names)

def config(filename=DEFAULT_CONFIG_FILENAME, section='postgresql'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
    db['options'] = '-c statement_timeout=360000000'
    return db

def test_connection(filename=DEFAULT_CONFIG_FILENAME):
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config(filename=filename)

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
        # create a cursor
        cur = conn.cursor()
        
    # execute a statement
        print('PostgreSQL database version:')
        cur.execute('SELECT version()')

        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        print(db_version)
       
        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')

def get_shared_buffers_size(config_filename = 'config/general_connector.ini'):
    sql = """select cast(setting as numeric) * 8192/(1024*1024) as shared_buffer_size from  
    pg_settings where name='shared_buffers';"""

    conn = establish_postgis_connection(config_filename=config_filename)
    cur = conn.cursor()
    cur.execute(sql)
    shared_buffer_size = float(cur.fetchall()[0][0])
    cur.close()
    conn.close()
    return shared_buffer_size
            
def establish_postgis_connection(config_filename = DEFAULT_CONFIG_FILENAME):
    params = config(filename=config_filename)
    conn = psycopg2.connect(**params)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    return conn



# retrieve POI by osm_id
def get_poi_by_osmid(conn, osm_id):
    sql = f'SELECT * FROM pois WHERE osm_id = {osm_id} LIMIT 3'
    result = GeoDataFrame.from_postgis(sql, conn, geom_col='geometry', 
        crs=None, index_col=None, coerce_float=True, parse_dates=None, params=None)
    return result


def get_greedy_search_path_by_keywords_frequencies(conn, sp, alternated=True, debug = False):
    sp_keywords = [v.keyword for v in sp.vertices]
    keyword_frequencies = get_keywords_frequencies(conn, sp_keywords)
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

def condition_for_keyword(column_names:list, keyword:str, boolean_connector:str = 'OR'):
    #boolean_connector must be either 'AND' or 'OR'
    expression = f"{column_names[0]} = '{keyword}' "
    for column in column_names[1:]:
        expression += f"{boolean_connector} {column} = '{keyword}' "
    return expression



# SIMPLIFIED RADIUS SEARCH
def search_by_keyword_and_radius(conn, keyword:str, center: tuple, radius: float, column_names, limit = None):
    lng, lat = center
    if limit is None:
        limit_check = ''
    else:
        limit_check = f'LIMIT {limit}'
    sql = f"""
    WITH tb1 AS
    (SELECT osm_id, ST_DistanceSphere(centroid, ST_Point({lng}, {lat}, 4326)) as distance
    FROM pois
    WHERE ST_DWithin(centroid::geography, ST_Point({lng}, {lat}, 4326)::geography, {radius}, false)
    AND ({condition_for_keyword(column_names, keyword)})
    )
    SELECT * FROM tb1 
    WHERE tb1.distance < {radius}
    {limit_check}
    """
    cur = conn.cursor()
    cur.execute(sql)
    return cur.fetchall()

def keyword_exists_inside_circle(conn, keyword:str, center: tuple, radius: float, column_names:list):
    return len(search_by_keyword_and_radius(conn, keyword, center, radius, column_names, limit = 2)) > 0


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

def with_clause_temporary_table_keyword(keyword, column_names = ['amenity','shop','tourism']):
    return f"""    tb_{keyword} AS
    (SELECT * FROM pois WHERE {condition_for_keyword(column_names, keyword)})"""

def with_clause_temporary_tables_all_keywords(keywords, column_names = ['amenity','shop','tourism']):
    return 'WITH\n' + ',\n'.join([with_clause_temporary_table_keyword(keyword, column_names = column_names) for keyword in keywords])

def select_clause_all_keywords(keywords, use_alias = True, include_centroids = False):
    temporary_table_names = [f'tb_{k}' for k in keywords]

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
    temporary_table_names = [f'tb_{k}' for k in keywords]
    return 'FROM ' + ', '.join(temporary_table_names)


def condition_clause_for_edge(edge):
    wi = edge.vi.keyword
    wj = edge.vj.keyword
    tb_vi_name = f'tb_{wi}'
    tb_vj_name = f'tb_{wj}'
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
    

def build_sql_query_for_spatial_pattern(sp, vertices_sequence_greedy, column_names, limit = 0):    
    if limit > 0:
        limit_check = f'LIMIT {limit}'
    else:
        limit_check = ''
    keywords_sequence_greedy = [v.keyword for v in vertices_sequence_greedy]
    keywords_sequence_in_sp = [v.keyword for v in sp.vertices]

    sql = f"""{with_clause_temporary_tables_all_keywords(keywords_sequence_greedy, column_names = column_names)}
    \r{select_clause_all_keywords(keywords_sequence_in_sp)}
    \rFROM tb_{keywords_sequence_greedy[0]}
    """
    for i, vertex in enumerate(vertices_sequence_greedy[1:]):
        keyword = vertex.keyword
        sql += f"""\rINNER JOIN tb_{keyword}
        \rON
        \r{condition_clause_for_multiple_edges(edges_of_vertex_and_previous(vertex, vertices_sequence_greedy[:i+1], sp))}
        """
    sql += f"""{limit_check}"""

    return sql    

def QQSPM_SQL(sp, connection = None, config_filename = DEFAULT_CONFIG_FILENAME, alternated=True, debug = False):
    global conn
    global COLUMN_NAMES
    global current_config_filename

    import resource_monitoring

    if connection is not None:
        conn = connection
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        COLUMN_NAMES = get_keyword_columns(conn)
    else:
        conn = establish_postgis_connection(config_filename = config_filename)
        if current_config_filename != config_filename:
            current_config_filename = config_filename
            COLUMN_NAMES = get_keyword_columns(conn)
    t0 = time()
    if debug:
        print(f'Column names set: {COLUMN_NAMES}')
    vertices_sequence_greedy = get_greedy_search_path_by_keywords_frequencies(conn, sp, alternated=alternated, debug = debug)
    if debug:
        print('Vertices sequence for Join:', vertices_sequence_greedy)
    sql_query = build_sql_query_for_spatial_pattern(sp, vertices_sequence_greedy, column_names = COLUMN_NAMES, limit = 0)
    if debug:
        print('SQL query:')
        print(sql_query)
    with conn.cursor() as cur:
        cur.execute(sql_query)
        conn.commit()
        solutions = cur.fetchall()
    conn.commit()
    memory_usage = resource_monitoring.get_total_memory_usage_for_postgres()
    conn.close()
    if debug == True:
        print('Total solutions:', len(solutions))
    elapsed_time = time() - t0
    return solutions, elapsed_time, memory_usage




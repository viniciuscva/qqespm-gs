#import __init__
import psycopg2
from geopandas import GeoDataFrame
from time import time
import psutil
from configparser import ConfigParser
from itertools import chain
import qqespm_quadtree_CGA as qq
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

import resource_monitoring

# this module creates postgis queries with implicit joins (where clause/ no specified order)

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

DEFAULT_CONFIG_FILENAME = 'config/london_pois_bbox.ini'
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
            
def establish_postgis_connection(config_filename = DEFAULT_CONFIG_FILENAME):
    params = config(filename=config_filename)
    conn = psycopg2.connect(**params)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    #conn.autocommit = True
    return conn



# retrieve POI by osm_id
def get_poi_by_osmid(conn, osm_id):
    sql = f'SELECT * FROM pois WHERE osm_id = {osm_id} LIMIT 3'
    result = GeoDataFrame.from_postgis(sql, conn, geom_col='geometry', 
        crs=None, index_col=None, coerce_float=True, parse_dates=None, params=None)
    return result

def get_all_osmids_from_db_conn(conn):
    cur = conn.cursor()
    cur.execute('SELECT DISTINCT(osm_id) FROM pois')
    osmids_sql = cur.fetchall()
    osmids_sql = list(list(zip(*osmids_sql))[0])
    cur.close()
    return osmids_sql

def condition_for_keyword(column_names:list, keyword:str, boolean_connector:str = 'OR'):
    #boolean_connector must be either 'AND' or 'OR'
    expression = f"{column_names[0]} = '{keyword}' "
    for column in column_names[1:]:
        expression += f"{boolean_connector} {column} = '{keyword}' "
    return expression

# SIMPLIFIED RADIUS SEARCH
def search_by_keyword_and_radius(conn, keyword:str, center: tuple, radius: float, column_names, approximation_buffer = 7, limit = None):
    lng, lat = center
    if limit is None:
        limit_check = ''
    else:
        limit_check = f'LIMIT {limit}'
    # obs: WEIRD BEHAVIOR WITH ST_DWITHIN, not find objects within radius when they exist, we have to expand radius to find them (+7)
    sql = f"""
    WITH tb1 AS
    (SELECT osm_id, ST_DistanceSphere(centroid, ST_Point({lng}, {lat}, 4326)) as distance
    FROM pois
    WHERE ST_DWithin(centroid::geography, ST_Point({lng}, {lat}, 4326)::geography, {radius+approximation_buffer}, false)
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

def build_exclusion_check(lij, sign, tb_vi_name, tb_vj_name):
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

def select_clause_all_keywords(keywords):
    temporary_table_names = [f'tb_{k}' for k in keywords]
    return 'SELECT ' + ', '.join([f'{ttn}.osm_id AS {ttn}_id' for ttn in temporary_table_names])

def from_clause_all_keywords(keywords):
    temporary_table_names = [f'tb_{k}' for k in keywords]
    return 'FROM ' + ', '.join(temporary_table_names)

def condition_clause_for_edge(edge, approximation_buffer = 7):
    wi = edge.vi.keyword
    wj = edge.vj.keyword
    tb_vi_name = f'tb_{wi}'
    tb_vj_name = f'tb_{wj}'
    lij, uij = edge.constraint['lij'], edge.constraint['uij']
    sign = edge.constraint['sign']
    relation = edge.constraint['relation']
    if lij > 0 and uij < float('inf'):
        distance_check = f'ST_DistanceSphere({tb_vi_name}.centroid, {tb_vj_name}.centroid) BETWEEN {lij} AND {uij} '
    elif lij == 0 and uij < float('inf'):
        distance_check = f'ST_DistanceSphere({tb_vi_name}.centroid, {tb_vj_name}.centroid) <= {uij} '
    elif lij > 0 and uij == float('inf'):
        distance_check = f'ST_DistanceSphere({tb_vi_name}.centroid, {tb_vj_name}.centroid) >= {lij} '
    else: # then lij == 0 and uij == float('inf')
        distance_check = ''
    if edge.constraint['is_exclusive']:
        exclusion_check = build_exclusion_check(lij, sign, tb_vi_name, tb_vj_name)
    else:
        exclusion_check = ''
    if relation is not None:
        relation_check = relations_to_postgis_functions[relation] + f'({tb_vi_name}.geometry, {tb_vj_name}.geometry)'
    else:
        relation_check = ''
    return ' AND \n'.join([check for check in [distance_check, exclusion_check, relation_check] if check != ''])

def condition_clause_for_all_edges(edges):
    return ' AND \n'.join([condition_clause_for_edge(e) for e in edges])

def build_sql_query_for_spatial_pattern(sp, column_names, limit = 0):
    edges = sp.edges
    keywords = [v.keyword for v in sp.vertices]
    if limit > 0:
        limit_check = f'LIMIT {limit}'
    else:
        limit_check = ''
    condition_clause = '    ' + condition_clause_for_all_edges(edges).replace("\n", "\n    ")
    sql = f"""{with_clause_temporary_tables_all_keywords(keywords, column_names = column_names)}
    \r{select_clause_all_keywords(keywords)}
    \r{from_clause_all_keywords(keywords)}
    \rWHERE 
    \r{condition_clause}
    \r{limit_check}"""
    return sql    

def QQSPM_SQL(sp, connection = None, config_filename = DEFAULT_CONFIG_FILENAME, debug = False):
    global conn
    global COLUMN_NAMES
    global current_config_filename

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
    sql_query = build_sql_query_for_spatial_pattern(sp, column_names = COLUMN_NAMES, limit = 0)
    #cur = conn.cursor()
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



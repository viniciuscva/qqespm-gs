DROP FUNCTION IF EXISTS distance_constraint;
CREATE OR REPLACE FUNCTION distance_constraint(keyword1 text, keyword2 text, min_distance float, max_distance float,
    first_excludes_second boolean, second_excludes_first boolean)
  RETURNS text
AS $$
import sys
sys.path.append('/home/vinicius/.local/lib/python3.11/site-packages/')
sys.path.append('/home/vinicius/Documents/qqespm_implementation')
import spatial_patterns_util as sps
import json

if min_distance<0 or max_distance<=min_distance:
	raise Exception('Invalid min or max distance')
	return None
if keyword1 == keyword2:
	raise Exception('Invalid pair of keywords. They must be different from each other')
	return None
if first_excludes_second and second_excludes_first:
	exclusion_sign = '<>'
elif first_excludes_second:
	exclusion_sign = '>'
elif second_excludes_first:
	exclusion_sign = '<'
else:
	exclusion_sign = '-'
return json.dumps({
	'keyword1': keyword1,
	'keyword2': keyword2,
	'min_distance': min_distance,
	'max_distance': max_distance,
	'exclusion_sign': exclusion_sign
})
$$ LANGUAGE plpython3u;


DROP FUNCTION IF EXISTS connectivity_constraint;
CREATE OR REPLACE FUNCTION connectivity_constraint(keyword1 text, keyword2 text, topological_relation text)
  RETURNS text
AS $$
import sys
sys.path.append('/home/vinicius/.local/lib/python3.11/site-packages/')
sys.path.append('/home/vinicius/Documents/qqespm_implementation')
import spatial_patterns_util as sps
import json

if not topological_relation in ['intersects', 'contains', 'within']:
	raise Exception('Invalid topological relation. Choose one among (intersects,contains,within)')
	return None
return json.dumps({
	'keyword1': keyword1,
	'keyword2': keyword2,
	'topological_relation': topological_relation,
})
$$ LANGUAGE plpython3u;

DROP FUNCTION IF EXISTS get_keywords_columns;
CREATE OR REPLACE FUNCTION get_keywords_columns(pois_table_name text)
  RETURNS text
AS $$
import sys
from itertools import chain
sys.path.append('/home/vinicius/.local/lib/python3.11/site-packages/')
sys.path.append('/home/vinicius/Documents/qqespm_implementation')
import spatial_patterns_util as sps
import json

keywords_columns = plpy.execute(f"select column_name from information_schema.columns where table_name = '{pois_table_name}'")
keywords_columns = [e['column_name'] for e in keywords_columns]
keywords_columns = list(set(keywords_columns) - {'osm_id', 'geometry', 'name', 'centroid', 'lon', 'lat', 'id'})
return json.dumps(keywords_columns)
$$ LANGUAGE plpython3u;

DROP FUNCTION IF EXISTS get_keywords_frequencies;
CREATE OR REPLACE FUNCTION get_keywords_frequencies(keywords text[], pois_table_name text, keyword_columns_json text)
  RETURNS text
AS $$
import sys
sys.path.append('/home/vinicius/.local/lib/python3.11/site-packages/')
sys.path.append('/home/vinicius/Documents/qqespm_implementation')
import spatial_patterns_util as sps
import json

keyword_columns = json.loads(keyword_columns_json)

keywords_frequencies = {}
for keyword in keywords:
	sql_query = f"""
	SELECT count(*)
	FROM {pois_table_name}
	WHERE ({sps.condition_for_keyword(keyword_columns, keyword)})
	"""
	keywords_frequencies[keyword] = plpy.execute(sql_query)[0]['count']
return json.dumps(keywords_frequencies)
$$ LANGUAGE plpython3u;

-- (_tb_school_id bigint, _tb_pharmacy_id bigint)


-- table (obj_keyword1 bigint,obj_keyword2 bigint,obj_keyword3 bigint,obj_keyword4 bigint, obj_keyword5 bigint) 
DROP FUNCTION IF EXISTS match_spatial_pattern;
CREATE OR REPLACE FUNCTION match_spatial_pattern(spatial_constraints text[], pois_table_name text, result_columns_order text[])
  RETURNS table (obj_keyword1 bigint,obj_keyword2 bigint,obj_keyword3 bigint,obj_keyword4 bigint, obj_keyword5 bigint) 
AS $$
import sys
import json
sys.path.append('/home/vinicius/.local/lib/python3.11/site-packages/')
sys.path.append('/home/vinicius/Documents/qqespm_implementation')
import spatial_patterns_util as sps

pois_table_name_exists = plpy.execute(f"select count(*) from information_schema.columns where table_name = '{pois_table_name}'")[0]['count'] > 0
if not pois_table_name_exists:
	raise Exception(f'The specified table name {pois_table_name} does not exist')
	return

sp = sps.spatial_pattern_from_constraints(spatial_constraints)
sp_keywords = [v.keyword for v in sp.vertices]
if set(result_columns_order) != set(sp_keywords):
	raise Exception('The names of the result columns in result_columns_order must match all the keywords in the spatial constraints')
	return
keywords_array_expression = "array["
for keyword in sp_keywords:
	keywords_array_expression += f"'{keyword}',"
keywords_array_expression = keywords_array_expression[:-1] + ']'
keywords_frequencies = plpy.execute(f"select * from get_keywords_frequencies({keywords_array_expression}, 'pois', get_keywords_columns('pois'))")[0]['get_keywords_frequencies']
	
vertices_order = sps.get_greedy_search_path_by_keywords_frequencies(sp, json.loads(keywords_frequencies), alternated = False)
keywords_columns = plpy.execute(f"select * from get_keywords_columns('{pois_table_name}')")[0]['get_keywords_columns']
keyword_column_names = json.loads(keywords_columns)
sql_query = sps.build_sql_query_for_spatial_pattern(sp, vertices_order, pois_table_name, keyword_column_names)
for record_ in plpy.execute(sql_query):
	next_row = {f"obj_keyword{i}": None for i in range(1,6)}
	for i, keyword in enumerate(result_columns_order):#sp_keywords
		next_row[f"obj_keyword{i+1}"] = record_[f"_tb_{keyword}_id"]
	yield next_row


$$ LANGUAGE plpython3u;

SELECT * FROM match_spatial_pattern(
	array[
		distance_constraint('school', 'pharmacy', 10, 1000, true, true) 
	], 
	'pois',
	array['pharmacy', 'school']
) limit 10;

-- select * from get_keywords_columns('pois');

--select * from get_keywords_frequencies(array['school', 'pharmacy'], 'pois', get_keywords_columns('pois'));


-- select * from distance_constraint('keyword1', 'keyword2', 0, 1500,
--                         true, true);

-- select * from connectivity_constraint('school', 'pharmacy', 'intersects')


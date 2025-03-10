#!/usr/bin/python3.8

import qqespm_quadtree_CGM as qqmin
import qqespm_quadtree_noc as qq1
import qqespm_quadtree_CGA as qq2
import espm_tv as qqsimple2
import qqespm_sql_imp as qqsql2
import qqespm_sql_exp as qqsql3
import qqespm_elastic as qqelastic3
import requests
import pandas as pd
from copy import deepcopy
import pickle
import gc
from time import sleep

dataset_file = 'london_pois_5500.csv'
ilquadtrees_dir = 'ilquadtrees_london_5500'
base_dataset_filename = 'data/london_pois_5500'
base_elastic_indexname = 'london_pois_5500_index'
base_postgresql_config_filename = 'config/london_pois_5500'

def writelog(string, file = 'log_comparison_london.txt'):
    with open(file, 'a') as f:
        f.write(string)

################################################################
# Keywords statistics
pois = qq2.read_df_csv(data_dir = f'data/{dataset_file}')

amenity_totals = pois.amenity.value_counts()
shop_totals = pois.shop.value_counts()
tourism_totals = pois.tourism.value_counts()
landuse_totals = pois.landuse.value_counts()
leisure_totals = pois.leisure.value_counts()
building_totals = pois.building.value_counts()


most_frequent_keywords = amenity_totals[amenity_totals>30].index.tolist() + \
    shop_totals[shop_totals>30].index.tolist() + \
    tourism_totals[tourism_totals>30].index.tolist() + \
    landuse_totals[landuse_totals>30].index.tolist() + \
    leisure_totals[leisure_totals>30].index.tolist() + \
    building_totals[building_totals>30].index.tolist()
most_frequent_keywords = list(set(most_frequent_keywords))

print('Size of dataset:', pois.shape[0])
writelog('Size of dataset:' + str(pois.shape[0]))
#print('Total most frequent keywords:', len(most_frequent_keywords))
#writelog('Total most frequent keywords:' + str(len(most_frequent_keywords)))
################################################################


################################################################
# Loading spatial patterns
with open('spatial_patterns_for_experiments_london.pkl', 'rb') as f:
    spatial_patterns = pickle.load(f)

print('Total spatial patterns:', len(spatial_patterns))
writelog('Total spatial patterns:' + str(len(spatial_patterns)))

executions_comparison = pd.DataFrame(columns = ['repetition', 'sp_index', 'spatial_pattern', 'number_of_vertices',
                                                'number_of_edges', 'number_of_exclusion_contraints', 'qualitative_prob', 'dataset_size', 'module', 
                                                'total_solutions', 'elapsed_time', 'memory_usage'])
################################################################


################################################################
# Preparing datasets information
percs = ['20perc','40perc','60perc','80perc','100perc']

datasets_info = {'QQ-Quadtree': {}, 'QQ-Elastic': {}, 'QQ-SQL': {}}
for perc in percs:
    datasets_info['QQ-Quadtree'][perc] = {'ilq_dir': f'{ilquadtrees_dir}/ilq_{perc}.pkl'}
    datasets_info['QQ-Elastic'][perc] = {'index_name': f'{base_elastic_indexname}_{perc}'}
    datasets_info['QQ-SQL'][perc] = {'config_filename': f'{base_postgresql_config_filename}_{perc}.ini'}

full_dataset_size = pois.shape[0]
dataset_sizes = [int(0.2*full_dataset_size), int(0.4*full_dataset_size), int(0.6*full_dataset_size), 
                 int(0.8*full_dataset_size), int(1*full_dataset_size)]
print('Datasets Info:', datasets_info)
writelog('Datasets Info: ' + str(datasets_info))

# ilqs = {}
# for perc in percs:
#     ilqs[perc] = qqsimple.generate_ram_ilquadtree(data_dir = f'{base_dataset_filename}_{perc}.csv', max_depth = 6, max_items = 500)
################################################################



################################################################
# Run the searches
num_repetitions = 1
for rep in range(1, num_repetitions+1):
    print('Starting repetition:', rep)
    writelog('Starting repetition:' + str(rep) + '\n')
    for sp_index, sp in enumerate(spatial_patterns):
        writelog(f'\n---------------------------------------------------------------\nPattern {sp_index} of {len(spatial_patterns)}:\n{sp.to_json()} repetition {rep} of {num_repetitions}\n')
        print('Pattern Number:', sp_index, ' out of', len(spatial_patterns), '/ Repetition:', rep, 'of', num_repetitions)
        for dataset_size_index, dataset_size in enumerate(dataset_sizes):
            perc = percs[dataset_size_index]
            print('Dataset size:' + str(dataset_size)+ ' (' + str(perc) + ')')
            writelog('Dataset size:' + str(dataset_size) + ' (' + str(perc) + ')\n')
            totals_of_solutions = []


            gc.collect()
            sleep(1)
            writelog('QQESPM-Quadtree_alternated\n')
            try:
                #return_values = func_timeout(10800, qq2.QQESPM, args=(deepcopy(sp),), kwargs=datasets_info['QQ-Quadtree'][perc])
                return_values = qq2.QQESPM(deepcopy(sp), **datasets_info['QQ-Quadtree'][perc])
                solutions_qq, elapsed_time_qq, memory_usage_qq = return_values
                total_solutions_qq = len(solutions_qq)
            # except FunctionTimedOut:
            #     writelog("Could not complete query within 10800 seconds and was terminated.\n")
            #     total_solutions_qq, elapsed_time_qq, memory_usage_qq = None, float('inf'), None
            except Exception as e:
                print("QQ-quadtree_alternated - Problem in pattern:" + str(e.__class__) + ' - ' + str(e) + '\n')
                writelog("QQ-quadtree_alternated - Problem in pattern:" + str(e.__class__) + ' - ' + str(e) + '\n')
                total_solutions_qq, elapsed_time_qq, memory_usage_qq = None, float('inf'), None
            row = [
                rep, 
                sp_index, 
                sp.to_json(),
                len(sp.vertices),
                len(sp.edges),
                sp.get_number_of_exclusion_contraints(),
                sp.qualitative_prob,
                dataset_size,
                'qq_quadtree_alternated',
                total_solutions_qq,
                elapsed_time_qq,
                memory_usage_qq
            ]
            executions_comparison.loc[len(executions_comparison)] = row
            executions_comparison.to_csv('executions_comparison_london.csv', mode = 'a', index = False, header = False)
            executions_comparison = executions_comparison[0:0]
            totals_of_solutions.append(total_solutions_qq)


            gc.collect()
            sleep(1)
            writelog('QQESPM-Quadtree_Min\n')
            try:
                #return_values = func_timeout(10800, qq2.QQESPM, args=(deepcopy(sp),), kwargs=datasets_info['QQ-Quadtree'][perc])
                return_values = qqmin.QQESPM(deepcopy(sp), **datasets_info['QQ-Quadtree'][perc])
                solutions_qq1, elapsed_time_qq1, memory_usage_qq1 = return_values
                total_solutions_qq1 = len(solutions_qq1)
            # except FunctionTimedOut:
            #     writelog("Could not complete query within 10800 seconds and was terminated.\n")
            #     total_solutions_qq1, elapsed_time_qq1, memory_usage_qq1 = None, float('inf'), None
            except Exception as e:
                print("QQMin - Problem in pattern:" + str(e.__class__) + ' - ' + str(e) + '\n')
                writelog("QQMin - Problem in pattern:" + str(e.__class__) + ' - ' + str(e) + '\n')
                total_solutions_qq1, elapsed_time_qq1, memory_usage_qq1 = None, float('inf'), None
            row = [
                rep, 
                sp_index, 
                sp.to_json(),
                len(sp.vertices),
                len(sp.edges),
                sp.get_number_of_exclusion_contraints(),
                sp.qualitative_prob,
                dataset_size,
                'qq_quadtree_min',
                total_solutions_qq1,
                elapsed_time_qq1,
                memory_usage_qq1
            ]
            executions_comparison.loc[len(executions_comparison)] = row
            executions_comparison.to_csv('executions_comparison_london.csv', mode = 'a', index = False, header = False)
            executions_comparison = executions_comparison[0:0]
            totals_of_solutions.append(total_solutions_qq1)


            gc.collect()
            sleep(1)
            writelog('QQESPM-Quadtree_OLD\n')
            try:
                #return_values = func_timeout(10800, qq2.QQESPM, args=(deepcopy(sp),), kwargs=datasets_info['QQ-Quadtree'][perc])
                return_values = qq1.QQESPM(deepcopy(sp), **datasets_info['QQ-Quadtree'][perc])
                solutions_qq, elapsed_time_qq, memory_usage_qq = return_values
                total_solutions_qq = len(solutions_qq)
            # except FunctionTimedOut:
            #     writelog("Could not complete query within 10800 seconds and was terminated.\n")
            #     total_solutions_qq, elapsed_time_qq, memory_usage_qq = None, float('inf'), None
            except Exception as e:
                print("QQ-quadtree_OLD - Problem in pattern:" + str(e.__class__) + ' - ' + str(e) + '\n')
                writelog("QQ-quadtree_OLD - Problem in pattern:" + str(e.__class__) + ' - ' + str(e) + '\n')
                total_solutions_qq, elapsed_time_qq, memory_usage_qq = None, float('inf'), None
            row = [
                rep, 
                sp_index, 
                sp.to_json(),
                len(sp.vertices),
                len(sp.edges),
                sp.get_number_of_exclusion_contraints(),
                sp.qualitative_prob,
                dataset_size,
                'qq_quadtree_old',
                total_solutions_qq,
                elapsed_time_qq,
                memory_usage_qq
            ]
            executions_comparison.loc[len(executions_comparison)] = row
            executions_comparison.to_csv('executions_comparison_london.csv', mode = 'a', index = False, header = False)
            executions_comparison = executions_comparison[0:0]
            totals_of_solutions.append(total_solutions_qq)



            gc.collect()
            sleep(1)
            writelog('QQ-simple\n')
            try:
                #return_values = qqsimple.QQ_SIMPLE(deepcopy(sp), ilqs[perc])
                return_values = qqsimple2.QQ_SIMPLE(deepcopy(sp), **datasets_info['QQ-Quadtree'][perc])
                solutions_qqs, elapsed_time_qqs, memory_usage_qqs = return_values
                total_solutions_qqs = len(solutions_qqs)
            # except FunctionTimedOut:
            #     writelog("Could not complete query within 10800 seconds and was terminated.\n")
            #     total_solutions_qqs, elapsed_time_qqs, memory_usage_qqs = None, float('inf'), None
            except Exception as e:
                print("QQ-simple - Problem in pattern:" + str(e.__class__) + ' - ' + str(e) + '\n')
                writelog("QQ-simple - Problem in pattern:" + str(e.__class__) + ' - ' + str(e) + '\n')
                total_solutions_qqs, elapsed_time_qqs, memory_usage_qqs = None, float('inf'), None
            row = [
                rep, 
                sp_index, 
                sp.to_json(),
                len(sp.vertices),
                len(sp.edges),
                sp.get_number_of_exclusion_contraints(),
                sp.qualitative_prob,
                dataset_size,
                'qq_simple',
                total_solutions_qqs,
                elapsed_time_qqs,
                memory_usage_qqs
            ]
            executions_comparison.loc[len(executions_comparison)] = row
            executions_comparison.to_csv('executions_comparison_london.csv', mode = 'a', index = False, header = False)
            executions_comparison = executions_comparison[0:0]
            totals_of_solutions.append(total_solutions_qqs)
        

            gc.collect()
            sleep(1)
            writelog('Elastic\n')
            try:
                #return_values = func_timeout(10800, qqelastic3.QQSPM_ELASTIC, args=(deepcopy(sp),), kwargs=datasets_info['QQ-Elastic'][perc])
                return_values = qqelastic3.QQSPM_ELASTIC(deepcopy(sp), **datasets_info['QQ-Elastic'][perc])
                solutions_el, elapsed_time_el, memory_usage_el = return_values
                total_solutions_el = len(solutions_el)
            # except FunctionTimedOut:
            #     writelog("Could not complete query within 10800 seconds and was terminated.\n")
            #     total_solutions_el, elapsed_time_el, memory_usage_el = None, float('inf'), None
            except Exception as e:
                print("Elastic - Problem in pattern:" + str(e.__class__) + ' - ' + str(e) + '\n')
                writelog("Elastic - Problem in pattern:" + str(e.__class__) + ' - ' + str(e) + '\n')
                total_solutions_el, elapsed_time_el, memory_usage_el = None, float('inf'), None
            row = [
                rep, 
                sp_index, 
                sp.to_json(),
                len(sp.vertices),
                len(sp.edges),
                sp.get_number_of_exclusion_contraints(),
                sp.qualitative_prob,
                dataset_size,
                'qq_elastic',
                total_solutions_el,
                elapsed_time_el,
                memory_usage_el
            ]
            executions_comparison.loc[len(executions_comparison)] = row
            executions_comparison.to_csv('executions_comparison_london.csv', mode = 'a', index = False, header = False)
            executions_comparison = executions_comparison[0:0]
            totals_of_solutions.append(total_solutions_el)



            gc.collect()
            sleep(1)
            writelog('SQL_implicit\n')
            try:
                #return_values = func_timeout(10800, qqsql2.QQSPM_SQL, args=(deepcopy(sp),), kwargs=datasets_info['QQ-SQL'][perc])
                return_values = qqsql2.QQSPM_SQL(deepcopy(sp), **datasets_info['QQ-SQL'][perc])
                solutions_sql, elapsed_time_sql, memory_usage_sql = return_values
                total_solutions_sql = len(solutions_sql)
            # except FunctionTimedOut:
            #     writelog("Could not complete query within 10800 seconds and was terminated.\n")
            #     total_solutions_sql, elapsed_time_sql, memory_usage_sql = None, float('inf'), None
            except Exception as e:
                print("SQL_implicit - Problem in pattern:" + str(e.__class__) + ' - ' + str(e) + '\n')
                writelog("SQL_implicit - Problem in pattern:" + str(e.__class__) + ' - ' + str(e) + '\n')
                total_solutions_sql, elapsed_time_sql, memory_usage_sql = None, float('inf'), None
            row = [
                rep, 
                sp_index, 
                sp.to_json(),
                len(sp.vertices),
                len(sp.edges),
                sp.get_number_of_exclusion_contraints(),
                sp.qualitative_prob,
                dataset_size,
                'qq_sql_implicit',
                total_solutions_sql,
                elapsed_time_sql,
                memory_usage_sql
            ]
            executions_comparison.loc[len(executions_comparison)] = row
            executions_comparison.to_csv('executions_comparison_london.csv', mode = 'a', index = False, header = False)
            executions_comparison = executions_comparison[0:0]
            totals_of_solutions.append(total_solutions_sql)
            



            gc.collect()
            sleep(1)
            writelog('SQL_explicit\n')
            try:
                #return_values = func_timeout(10800, qqsql2.QQSPM_SQL, args=(deepcopy(sp),), kwargs=datasets_info['QQ-SQL'][perc])
                return_values = qqsql3.QQSPM_SQL(deepcopy(sp), **datasets_info['QQ-SQL'][perc])
                solutions_sql, elapsed_time_sql, memory_usage_sql = return_values
                total_solutions_sql = len(solutions_sql)
            # except FunctionTimedOut:
            #     writelog("Could not complete query within 10800 seconds and was terminated.\n")
            #     total_solutions_sql, elapsed_time_sql, memory_usage_sql = None, float('inf'), None
            except Exception as e:
                print("SQL_explicit - Problem in pattern:" + str(e.__class__) + ' - ' + str(e) + '\n')
                writelog("SQL_explicit - Problem in pattern:" + str(e.__class__) + ' - ' + str(e) + '\n')
                total_solutions_sql, elapsed_time_sql, memory_usage_sql = None, float('inf'), None
            row = [
                rep, 
                sp_index, 
                sp.to_json(),
                len(sp.vertices),
                len(sp.edges),
                sp.get_number_of_exclusion_contraints(),
                sp.qualitative_prob,
                dataset_size,
                'qq_sql_explicit',
                total_solutions_sql,
                elapsed_time_sql,
                memory_usage_sql
            ]
            executions_comparison.loc[len(executions_comparison)] = row
            executions_comparison.to_csv('executions_comparison_london.csv', mode = 'a', index = False, header = False)
            executions_comparison = executions_comparison[0:0]
            totals_of_solutions.append(total_solutions_sql)

            

            if len(set(totals_of_solutions)) != 1:
                writelog(f'Divergence in pattern: {sp.to_json()}\nTotal solutions for qq_quadtree_alternated, qq_quadtree_min, qq_quadtree_old, qq_simple, qq_elastic, qq_sql_implicit, qq_sql_explicit: {totals_of_solutions}\n')
                print(f'Divergence in pattern: {sp.to_json()}\nTotal solutions for qq_quadtree_alternated, qq_quadtree_min, qq_quadtree_old, qq_simple, qq_elastic, qq_sql_implicit, qq_sql_explicit: {totals_of_solutions}\n')

################################################################

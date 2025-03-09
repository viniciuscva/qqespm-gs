# QQESPM-GS: A generalized solution for the QQESPM spatio-textual query approach

This repository contains the code implementations for the algorithms proposed in my Master's thesis. The algorithms aim at answering a complex type of geo-textual group query called [QQ-SPM query](https://arxiv.org/abs/2312.08992). The Quantitative and Qualitative Efficient Spatial Pattern Matching Generalized Solution (QQESPM-GS) is our proposed comprehensive approach for handling QQ-SPM queries efficiently across various geospatial technologies. QQESPM-GS consists of three primary solutions. 

1. QQESPM-Quadtree: an algorithm that utilizes the IL-Quadtree index, memoization techniques, and optimized join ordering to efficiently address QQ-SPM queries through a specialized approach. This library is an ad hoc solution and can be executed without spatial databases or backend GIR systems.
2. QQESPM-EO: an algorithm that manages the execution of elementary spatial operations in a Geographic Information Retrieval (GIR) system to effectively solve a QQ-SPM query. We showcase QQESPM-EO by using the Elasticsearch as the backend for the elementary spatio-textual operations, culminating in the QQESPM-Elastic library. This library requires Elasticsearch as a dependency.
3. QQESPM-SQL: a pipeline that translates spatio-textual requirements from a QQ-SPM graph into an efficient SQL spatial query that is then executed against a PostgreSQL database with the PostGIS spatial extension. This library requires PostgreSQL and PostGIS as dependencies.

## Reproducibility Guide

Having PostgreSQL with PostGIS and Elasticsearch locally installed is required for running the scripts. Also, the python libs dependencies are specified in `requirements.txt`.

Follow the steps in `data/README.md` file to get two base POIs datasets in CSV files (`london_pois_5500.csv` for Experiment 1 from Paper and `london_pois_bbox.csv` for Experiment 2 from Paper).


Create smaller subsets of the full datasets of POIs (for dataset scalability experiments) and index in ILQuadtrees on disk:
* Create the folder `ilquadtrees`; inside this folder, create the folders 20perc, 40perc, 60perc, 80perc, 100perc
* Follow instructions in `create_csv_and_ilq_subdatasets.ipynb` notebook
    * For Experiment 1, use `data_dir = 'data/london_pois_5500.csv'`, `ilq_base_folder = 'ilquadtrees_london_5500'` and `base_csv_filename = 'data/london_pois_5500'` in this notebook.
    * For Experiment 2, use `data_dir = 'data/london_pois_bbox.csv'`, `ilq_base_folder = 'ilquadtrees_london_pois_bbox'` and `base_csv_filename = 'data/london_pois_bbox'` in this notebook.


Index the datasets in Elasticsearch:
* Go to `elastic_module` folder in the project repository
* Follow instructions in `generate_geojson.ipynb notebook`
    * For Experiment 1, use `base_dataset_filename = 'london_pois_5500'` in this notebook
    * For Experiment 2, use `base_dataset_filename = 'london_pois_bbox'` in this notebook
* Follow instructions in `geojson_to_elasticsearch.ipynb` notebook
    * For Experiment 1, use `base_index_name = 'london_pois_5500_index'` and `base_dataset_filename = 'london_pois_5500'` in this notebook
    * For Experiment 2, use `base_index_name = 'london_pois_bbox_index'` and `base_dataset_filename = 'london_pois_bbox'` in this notebook.


Load datasets into tables in PostGIS database system:
* Follow instructions in notebook `create_dbs_postgres.ipynb`
    * For Experimen1 1, use `base_db_name = 'london_pois_5500'` and `base_csv_filename = 'london_pois_5500'` in this notebook
    * For Experiment 2, use `base_db_name = 'london_pois_bbox'` and `base_csv_filename = 'london_pois_bbox'` in this notebook.


Generate spatial patterns for queries:
* Follow instructions in `generate_spatial_patterns.ipynb` notebook
    * For Experiment 1, use `dataset_file = 'london_pois_5500_100perc.csv'` in this notebook
    * For Experiment 2, use `dataset_file = 'london_pois_bbox_100perc.csv'` in this notebook


Increase the max result window on Elasticsearch to a value greater than or equal to the dataset size.

It's recommended to set up `shared_buffers` parameter in postgresql.conf to 25% of the RAM size.


Start the experiments
* `python compare_modules.py`
    * For Experiment 1, set `dataset_file = 'london_pois_5500_100perc.csv'`, `ilquadtrees_dir = 'ilquadtrees_london_5500'`, `base_dataset_filename = 'data/london_pois_5500'`, `base_elastic_indexname = 'london_pois_5500_index'` and `base_postgresql_config_filename = 'config/london_pois_5500'` at the start of this script
    * For Experiment 2, set `dataset_file = 'london_pois_bbox_100perc.csv'`, `ilquadtrees_dir = 'ilquadtrees_london_pois_bbox'`, `base_dataset_filename = 'data/london_pois_bbox'`, `base_elastic_indexname = 'london_pois_bbox_index'` and `base_postgresql_config_filename = 'config/london_pois_bbox'` at the start of this script
* This script writes to files `log_comparison_london_new.txt` (a basic logs file) and `executions_comparison_london.csv` (a log of all query execution times along with their respective query configuration, thus useful for future performance analysis and visualization). 


Generate performance comparison visualizations:
* Follow instructions in notebook `comparing_modules.ipynb`



## PostgreSQL extension for QQESPM-SQL

We also implemented a PostgreSQL extension in PL/pgSQL to replicate the strategy of QQESPM-SQL internally within the native environment of the PostgreSQL database. The source-code for this extension is in [this repository](https://github.com/viniciuscva/qqespm_postgres_extension).

## License

These algorithms and implementation libraries of QQESPM-Quadtree, QQESPM-Elastic and QQESPM-SQL © 2024 by [Carlos Vinicius A. M. Pontes](https://www.linkedin.com/in/vinicius-alves-mm/) are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/?ref=chooser-v1).

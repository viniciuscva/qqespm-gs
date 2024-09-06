# QQESPM-GS: A generalized solution for the QQESPM spatio-textual query approach

This repository contains the code implementations for the algorithms proposed in my Master's thesis. The algorithms aim at answering a complex type of geo-textual group query called [QQ-SPM query](https://arxiv.org/abs/2312.08992). The Quantitative and Qualitative Efficient Spatial Pattern Matching Generalized Solution (QQESPM-GS) is our proposed comprehensive approach for handling QQ-SPM queries efficiently across various geospatial technologies. QQESPM-GS consists of three primary solutions. 

1. QQESPM-Quadtree: an algorithm that utilizes the IL-Quadtree index, memoization techniques, and optimized join ordering to efficiently address QQ-SPM queries through a specialized approach. This library is an ad hoc solution and can be executed without spatial databases or backend GIR systems.
2. QQESPM-EO: an algorithm that manages the execution of elementary spatial operations in a Geographic Information Retrieval (GIR) system to effectively solve a QQ-SPM query. We showcase QQESPM-EO by using the Elasticsearch as the backend for the elementary spatio-textual operations, culminating in the QQESPM-Elastic library. This library requires Elasticsearch as a dependency.
3. QQESPM-SQL: a pipeline that translates spatio-textual requirements from a QQ-SPM graph into an efficient SQL spatial query that is then executed against a PostgreSQL database with the PostGIS spatial extension. This library requires PostgreSQL and PostGIS as dependencies.

We also implemented a PostgreSQL extension in PL/pgSQL to replicate the strategy of QQESPM-SQL internally within the native environment of the PostgreSQL database. The source-code for this extension is in [this repository](https://github.com/viniciuscva/qqespm_postgres_extension).

## License

These algorithms and implementation libraries of QQESPM-Quadtree, QQESPM-Elastic and QQESPM-SQL © 2024 by [Carlos Vinicius A. M. Pontes](https://www.linkedin.com/in/vinicius-alves-mm/) are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/?ref=chooser-v1).

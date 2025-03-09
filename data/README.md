The experiments encompass running QQ-SPM queries on subsets of a dataset of POIs from London, UK, gathered from OpenStreetMap. The following steps guide on how to download a similar extraction from [OpenStreetMap](https://www.openstreetmap.org/) to reproduce experiments.

Download geographic features (including POIs data) from OpenStreetMap, for the [Greater London](https://download.geofabrik.de/europe/united-kingdom/england/greater-london.html) region. 


Upload the data to a PostgreSQL local database with the geospatial extension PostGIS using the [osm2pgsql](https://osm2pgsql.org/) cli tool:

```
osm2pgsql -H localhost -d osm_london_pois -U postgres -P 5432 -S default.style --hstore -W -F flatnode_osm2pgsql.cache --slim greater-london-latest.osm.pbf
```

Filter geographic features to only POIs having the tags amenity, shop, tourism, landuse, leisure, building, and geographically limited by bounding box extension by following steps in file `filter_pois_by_tag_and_bbox_region.sql`. 
After that you will ideally have two base POI datasets in CSV (`london_pois_5500.csv` and `london_pois_bbox.csv`), that are used in query performance experiments.

Follow the steps in notebook `prepare_london_dataset.ipynb` to add buffers to the polygonal geometries delimiting the POIs for both datasets.


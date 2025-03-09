CREATE TABLE  london_pois AS 
(SELECT osm_id, name, amenity, shop, tourism, landuse, leisure, building, way AS geometry, 'point' AS data_type FROM planet_osm_point  WHERE concat(amenity, shop, tourism, landuse, leisure, building) != '')
union
(SELECT osm_id, name, amenity, shop, tourism, landuse, leisure, building, way AS geometry, 'line' AS data_type FROM planet_osm_line  WHERE concat(amenity, shop, tourism, landuse, leisure, building) != '')
union
(SELECT osm_id, name, amenity, shop, tourism, landuse, leisure, building, way AS geometry, 'polygon' AS data_type FROM planet_osm_polygon  WHERE concat(amenity, shop, tourism, landuse, leisure, building) != '')
union
(SELECT osm_id, name, amenity, shop, tourism, landuse, leisure, building, way AS geometry, 'roads' AS data_type FROM planet_osm_roads  WHERE concat(amenity, shop, tourism, landuse, leisure, building) != '')
;

UPDATE london_pois SET geometry = ST_TRANSFORM(geometry, 4326);
CREATE INDEX spatial_index_london_pois ON london_pois USING GIST ( geometry );
CREATE INDEX spatial_index_sp_london_pois ON london_pois USING SPGIST ( geometry );

-- Create a dataset only with the POIs within a squared lat-long bounding box of 5500m of both horizontal and vertical extension centered on London central coordinates (-0.118092, 51.509865):
CREATE TABLE pois_5500m AS
SELECT *
FROM   london_pois
WHERE  geometry @ 
    ST_MakeEnvelope(-0.15782873513750623,
 51.48513365583724,
 -0.07835526486445585,
 51.534596344162765);

-- Create a dataset only with the POIs within a squared lat-long bounding box of 12000m of both horizontal and vertical extension centered on London central coordinates (-0.118092, 51.509865):
CREATE TABLE pois_12000m AS
SELECT *
FROM   london_pois
WHERE  geometry @ 
    ST_MakeEnvelope(-0.20479033521186382,
 51.45590570364487,
 -0.03139366479178516,
 51.56382429635512)

-- Download the CSV data from these tables to get the base datasets for the experiments:

COPY (SELECT  osm_id, name, amenity, shop, tourism, landuse, leisure, building, data_type, ST_AsText(geometry) as geometry from pois_5500m) TO 'london_pois_5500.csv'  WITH DELIMITER ';' CSV HEADER;

COPY (SELECT  osm_id, name, amenity, shop, tourism, landuse, leisure, building, data_type, ST_AsText(geometry) as geometry from pois_12000m) TO 'london_pois_bbox.csv'  WITH DELIMITER ';' CSV HEADER;

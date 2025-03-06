WITH
    tb_waste_basket AS
    (SELECT * FROM pois WHERE amenity = 'waste_basket' OR building = 'waste_basket' OR 
    landuse = 'waste_basket' OR leisure = 'waste_basket' OR shop = 'waste_basket' OR tourism = 'waste_basket' ),
    tb_bicycle AS
    (SELECT * FROM pois WHERE amenity = 'bicycle' OR building = 'bicycle' OR 
    landuse = 'bicycle' OR leisure = 'bicycle' OR shop = 'bicycle' OR tourism = 'bicycle' ),
    tb_travel_agency AS
    (SELECT * FROM pois WHERE amenity = 'travel_agency' OR building = 'travel_agency' OR 
    landuse = 'travel_agency' OR leisure = 'travel_agency' OR shop = 'travel_agency' OR tourism = 'travel_agency' )
SELECT tb_waste_basket.osm_id AS tb_waste_basket_id, tb_bicycle.osm_id AS tb_bicycle_id, 
tb_travel_agency.osm_id AS tb_travel_agency_id
FROM tb_waste_basket, tb_bicycle, tb_travel_agency
WHERE 
    ST_CoveredBy(tb_waste_basket.geometry, tb_bicycle.geometry) AND 
    ST_DistanceSphere(tb_bicycle.centroid, tb_travel_agency.centroid) BETWEEN 81.36 AND 367.12  AND 
    NOT EXISTS (SELECT 1 FROM tb_bicycle aux WHERE 
        ST_DWithin(tb_travel_agency.centroid::geography, aux.centroid::geography, 81.36, false)) 

import geojson
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas
import shapely
from shapely.geometry import shape
#from toporel import compute_relation, compute_all_relations
from lat_lon_distance2 import lat_lon_distance
import pandas as pd
import json

class hashabledict(dict):
    def __hash__(self):
        return hash(json.dumps(self))

class GeoObj:
    def __init__(self, item, conn = None):
        self.item = item
        self.conn = conn
        
        if type(item) == dict or type(item) == hashabledict: # the item dict must come with 'geometry' and 'keywords' entries
            #self.item['geometry'] = shape(geometry)
            #self.item['keywords'] = keywords
            #self.item['geometry_type'] = self.item['geometry'].geom_type
            #self.bounds = self.item['geometry'].bounds
            self.item['centroid'] = self.item['geometry'].centroid.coords[0]
            #self.x, self.y = self.item['centroid'].coords[0]
            self.item['bbox'] = self.item['geometry'].bounds
            self.conn = conn
        elif type(item) == pd.Series:
            # we expect self.item['centroid'] already exists as a column
            # we expect self.item['bbox'] already exists as a column
            pass
            
    def __str__(self):
        return 'GeoObj /item: ' + str(self.item) + ' /geometry: ' + str(self.geometry) + ' /keywords: ' + str(self.keywords)
    
    def __hash__(self):
        return hash(json.dumps(self.get_data()))
    
    def __eq__(self, another):
        return self.item['osm_id'] == another.item['osm_id']

    def geometry(self):
        if type(self.item) == dict or type(self.item) == pd.Series or type(self.item) == hashabledict:
            return self.item['geometry']
        else:
            # TODO
            # CONNECTION TO DB
            pass

    def centroid(self):
        if type(self.item) == dict or type(self.item) == pd.Series or type(self.item) == hashabledict:
            return self.item['centroid']
        else:
            # TODO
            # CONNECTION TO DB
            pass

    def keywords(self):
        if type(self.item) == dict or type(self.item) == hashabledict:
            return self.item['keywords']
        elif type(self.item) == pd.Series:
            return set(self.item[['amenity', 'shop', 'tourism', 'landuse', 'leisure', 'building', 'office', 'government']].values) - {''}
        else:
            # TODO
            # CONNECTION TO DB
            pass

    def bbox(self):
        if type(self.item) == dict or type(self.item) == pd.Series or type(self.item) == hashabledict:
            return self.item['bbox']
        else:
            # TODO
            # CONNECTION TO DB
            pass
        
    def distance(self, another_geoobj, metric = 'geodesic'):
        #return self.geometry.distance(another_geoobj.geometry)
        lonA, latA = self.centroid()
        lonB, latB = another_geoobj.centroid()
        return lat_lon_distance(latA, lonA, latB, lonB, metric)
        #return distance(lonA, latA, lonB, latB)
    
    def get_data(self):
        return {'id': self.item.get('osm_id'),
                'location': self.item.get('centroid'),
                'description': str(self.item.get('name')) + ' (' + ','.join(self.keywords()) + ')'}
    
    def get_description(self):
        return {'id': self.item.get('id'),
                'osmid': self.item.get('osmid'),
                'name': self.item.get('name')}
                
    
    # def relations_with(self, another_geoobj):
    #     return compute_all_relations(self.geometry(), another_geoobj.geometry())
    
    def relation(self, another_geoobj):
        #return compute_relation(self.geometry(), another_geoobj.geometry())
        if self.intersects(another_geoobj):
            if self.geometry().contains(another_geoobj.geometry()):
                return 'contains'
            elif another_geoobj.geometry().contains(self.geometry()):
                return 'within'
            else:
                return 'intersects'
        else:
            return 'disjoint'
    
    def intersects(self, another_geoobj):
        return self.geometry().intersects(another_geoobj.geometry())
        
    @staticmethod
    def get_objects_from_geopandas(gdf, keyword_columns = ['amenity','shop','tourism','landuse','leisure','building','office','government']):

        def str_or_None(x):
            if type(x)==str:
                return x
            return None
        
        geoobjs = []
        for i, row in gdf.iterrows():
            keywords = [str_or_None(row[column]) for column in keyword_columns]
            keywords = set(keywords) - {None}
            
            if 'osm_id' in row:
                id_value = row['osm_id']
            elif 'id' in row:
                id_value = row['id']
            elif 'osmid' in row:
                id_value = row['osmid']
            else:
                id_value = ''

            item = {'osm_id': id_value, #hashabledict(**
                    'name': row['name'],
                    'geometry': row['geometry'],
                    'category': '',
                    'keywords': keywords}#)

            for column in keyword_columns:
                if type(row[column]) == str:
                    item['category'] += f'{column},'
            # else:
            #     item['category'] = 'undefined'
            geoobj = GeoObj(item)
            geoobjs.append(geoobj)
        return geoobjs
    
    
    @staticmethod
    def get_objects_from_geojson_fc(feature_collection):
        geoobjs = []
        for i,feature in enumerate(feature_collection.features):
            if ('amenity' not in feature.properties) and ('shop' not in feature.properties) and ('name' not in feature.properties): 
                continue
            geometry = shape(feature.geometry)
            keywords = [feature.properties.get('amenity'), feature.properties.get('shop'), feature.properties.get('name')]
            if 'id' in feature.keys():
                item = {'id': feature.id}
            elif 'osmid' in feature.properties:
                item = {'osmid': feature.properties['osmid']}
            else:
                item = feature
            geoobj = GeoObj(item, keywords, geometry)
            geoobjs.append(geoobj)
        return geoobjs
    
    
    @staticmethod
    def get_objects_from_geojson_file(geojson_file):
        feature_collection = geojson_fc_from_file(geojson_file)
        return GeoObj.get_objects_from_geojson_fc(feature_collection)
    
    def to_geojson_feature(self):
        geometry = geojson.loads(shapely.to_geojson(self.geometry))
        feature = geojson.Feature(geometry=geometry, 
                          properties={'amenity': self.keywords[0],
                                      'element_type': self.item['element_type'],
                                      'name': self.item['name'],
                                      'osmid': self.item['osmid']}) 
        return feature
    

    def to_json(self):
        return geojson.dumps(self.to_geojson_feature())


def get_ndim_of_geometry(geom):
    geom_type = geom.geometry_type
    if geom_type in ['Point', 'MultiPoint']: return 0
    elif geom_type in ['LineString', 'MultiLineString']: return 1
    else: return 2
    
def gdf_from_geojson_file(geojson_file):
    with open(geojson_file) as f:
        feature_collection = geojson.load(f)
    return geopandas.GeoDataFrame.from_features(feature_collection)

def save_gdf_to_geojson_file(gdf, output_file):
    return gdf.to_file(output_file, drive = 'GeoJSON')

def geojson_fc_from_file(geojson_file):
    with open(geojson_file) as f:
        features = geojson.load(f)
    return features

def save_geojson_fc_to_file(geojson_features, output_file):
    with open(output_file, 'w') as f:
        geojson.dump(geojson_features, f)
        
def gdf_to_geojson_fc(gdf):
    feature_collection = geojson.loads(gdf.to_json())
    return feature_collection

def geojson_fc_to_gdf(feature_collection):
    return geopandas.GeoDataFrame.from_features(feature_collection)

def geoseries_to_geoobj(row):
    geometry = row['geometry']
    keywords = [str_or_None(row['amenity']), str_or_None(row['shop']), str_or_None(row['tourism']),
               str_or_None(row['landuse']), str_or_None(row['leisure']), str_or_None(row['building']),
               str_or_None(row['office']), str_or_None(row['government'])]
    keywords = set(keywords) - {None}
    if 'osm_id' in row:
        id_value = row['osm_id']
    elif 'id' in row:
        id_value = row['id']
    elif 'osmid' in row:
        id_value = row['osmid']
    else:
        id_value = ''
    item = {'osmid': id_value,
            'category': ''}#,
            #'name': row['name'],}
    
    if type(row['amenity']) == str:
        item['category'] += 'amenity,'
    if type(row['shop']) == str:
        item['category'] += 'shop,'
    if type(row['tourism']) == str:
        item['category'] += 'tourism,'
    if type(row['landuse']) == str:
        item['category'] += 'landuse,'
    if type(row['leisure']) == str:
        item['category'] += 'leisure,'
    if type(row['building']) == str:
        item['category'] += 'building,'
    if type(row['office']) == str:
        item['category'] += 'office,'
    if type(row['government']) == str:
        item['category'] += 'government,'
    # else:
    #     item['category'] = 'undefined'
    try:
        geoobj = GeoObj(item, keywords, geometry)
    except:
        geoobj = None
    return geoobj
    #geoobjs_lst.append(geoobj)

def str_or_None(x):
    if type(x)==str:
        return x
    return None
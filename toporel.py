from shapely import relate, relate_pattern, Point, Polygon


relate_patterns = {
    'equals': ['T*F**FFF*'],
    'disjoint': ['FF*FF****'],
    'touches': ['FT*******', 'F**T*****', 'F***T****'],
    'contains': ['T*****FF*'],
    'covers': ['T*****FF*', '*T****FF*', '***T**FF*', '****T*FF*'],
    'intersects': ['T********', '*T*******', '***T*****', '****T****'],
    'within': ['T*F**F***'],
    'coveredby': ['T*F**F***', '*TF**F***', '**FT*F***', '**F*TF***'],
    'crosses': ['T*T******', 'T*****T**', '0********'],
    'overlaps': ['T*T***T**', '1*T***T**']
}

def get_ndim_of_geometry(geom):
    geom_type = geom.geometry_type
    if geom_type in ['Point', 'MultiPoint']: return 0
    elif geom_type in ['LineString', 'MultiLineString']: return 1
    else: return 2

def compute_relation(geomA, geomB):
    rel_AB = geomA.relate(geomB)
    if any([one_toporel_satisfies_the_other(rel_AB, r) for r in relate_patterns['covers']]):
        return 'contains'
    if any([one_toporel_satisfies_the_other(rel_AB, r) for r in relate_patterns['coveredby']]):
        return 'within'
    if any([one_toporel_satisfies_the_other(rel_AB, r) for r in relate_patterns['intersects']]):
        return 'intersects'
    if any([one_toporel_satisfies_the_other(rel_AB, r) for r in relate_patterns['disjoint']]):
        return 'disjoint'
    
def compute_all_relations(geomA, geomB):
    rel_AB = geomA.relate(geomB)
    all_relations = []
    if any([one_toporel_satisfies_the_other(rel_AB, r) for r in relate_patterns['covers']]):
        all_relations.append('contains')
    if any([one_toporel_satisfies_the_other(rel_AB, r) for r in relate_patterns['coveredby']]):
        all_relations.append('within')
    if any([one_toporel_satisfies_the_other(rel_AB, r) for r in relate_patterns['disjoint']]):
        all_relations.append('disjoint')
    if any([one_toporel_satisfies_the_other(rel_AB, r) for r in relate_patterns['intersects']]):
        all_relations.append('intersects')
    return all_relations

def get_average_side_length(poly):
    if poly.geom_type == 'Point':
        return 0
    return poly.length/len(set(poly.exterior.coords))

def apply_default_buffer(poly, fraction = 0.10, point_buffer_size = 0.00034395064773051537):# the mean of all the average side lengths for the polygons in the dataset
    poly = poly.convex_hull.simplify(0.05)
    if poly.geom_type == 'Point':
        buffer_size = point_buffer_size
    else:
        buffer_size = fraction * get_average_side_length(poly)
    poly = poly.buffer(buffer_size, quad_segs = 1)
    return poly


def generalize_dimension(dimension):
    # dimension is a single character, representing the size of an 2D intersection in the DE9IM model
    dimension = str(dimension)
    if dimension in ['0','1','2']: return 'T' # True
    elif dimension == '-1': return 'F' # False
    return dimension

def is_generalized_dimension_size(dimension):
    # dimension is a single character, representing the size of an 2D intersection in the DE9IM model
    return dimension in ['T', 'F']

def is_wildcard_dimension_size(dimension):
    return dimension == '*'

def one_toporel_satisfies_the_other(toporel_a, toporel_b):
    # toporel_a and toporel_b are DE9IM masks, which are strings like 'F**T*****', for example, each with 9 characters
    toporel_a = list(toporel_a)
    toporel_b = list(toporel_b)
    for i in range(len(toporel_a)):
        if is_generalized_dimension_size(toporel_b[i]):
            toporel_a[i] = generalize_dimension(toporel_a[i])
        if is_wildcard_dimension_size(toporel_b[i]):
            toporel_a[i] = toporel_b[i]
    return toporel_a == toporel_b

def one_toporel_satisfies_some_other(toporel_a, toporel_list):
    for tr in toporel_list:
        if one_toporel_satisfies_the_other(toporel_a, tr): return True
    return False

def get_administrative_boundaries():
    import pandas as pd
    from shapely import wkt
    import os
    os.environ['USE_PYGEOS'] = '0'
    import geopandas

    regions = pd.read_csv('data/administrative_boundaries.csv')
    regions['geometry'] = regions['geometry'].apply(wkt.loads)
    regions = geopandas.GeoDataFrame(regions)
    return regions

def get_covering_regions(geom, regions = None):
    #from toporel import rel_coveredby
    if regions is None:
        regions = get_administrative_boundaries()
    covering_regions = []
    for i,row in regions.iterrows():
        if rel_coveredby(geom, row.geometry):
            covering_regions.append(( row['name'], row.geometry))
    return covering_regions

def municipality(geom, regions = None):
    if regions is None:
        regions = get_administrative_boundaries()
    regions = get_covering_regions(geom, regions)
    regions.sort(key = lambda e: e[1].envelope.area)
    return regions[1][0]

#check https://en.wikipedia.org/wiki/DE-9IM for more info about the DE-9IM topological relation model.


#font: https://pt.martech.zone/calculate-great-circle-distance/
#from numpy import sin, cos, arccos, pi, round
from math import sin, cos, asin, acos, sqrt, pi

# def rad2deg(radians):
#     degrees = radians * 180 / pi
#     return degrees

# def deg2rad(degrees):
#     radians = degrees * pi / 180
#     return radians

# def lat_lon_distance(latitude1, longitude1, latitude2, longitude2, unit = 'miles'):
    
#     theta = longitude1 - longitude2
    
#     distance = 60 * 1.1515 * rad2deg(
#         arccos(
#             (sin(deg2rad(xA)) * sin(deg2rad(xB))) + 
#             (cos(deg2rad(xA)) * cos(deg2rad(xB)) * cos(deg2rad(theta)))
#         )
#     )
    
#     if unit == 'miles':
#         return round(distance, 2)
#     elif unit == 'kilometers':
#         return round(distance * 1.609344, 2)


def lat_lon_distance(lat1, lon1, lat2, lon2, metric = 'geodesic'):
    if metric == 'geodesic':
        r = 6371 # km
        p = pi / 180

        a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
        return 2000 * r * asin(sqrt(a))
    elif metric == 'euclidian':
        return ((lat1-lat2)**2 + (lon1-lon2)**2)**0.5



def shape_distance(shapeA, shapeB, metric = 'geodesic'):
    lonA, latA = shapeA.centroid.coords[0]
    lonB, latB = shapeB.centroid.coords[0]
    return lat_lon_distance(latA, lonA, latB, lonB)


def get_horizontal_extremes(center, dist):
    """
    Given a center = (long, lat), and a distance dist in meters, finds the longitudes lon1 and lon2
    such that the locations (lon1, y) and (lon2, y) are dist meters away from the center
    """
    x, y = center # lng, lat
    a = cos(x*pi/180.0)
    b = sin(x*pi/180.0)
    k = 2*(sin(dist/12742000.0)/cos(y*pi/180.0))**2
    n1 = -a*(k-1) + b*sqrt(k*(2-k))
    n2 = -a*(k-1) - b*sqrt(k*(2-k))
    lon1 = 180*acos(n1)/pi
    if abs(lat_lon_distance(y, x, y, lon1) - dist) > 1:
        lon1 = -lon1
        
    lon2 = 180*acos(n2)/pi
    if abs(lat_lon_distance(y, x, y, lon2) - dist) > 1:
        lon2 = -lon2

    if lon1 > lon2:
        lon1, lon2 = lon2, lon1
    return [lon1, lon2]

def get_vertical_extremes(center, dist):
    """
    Given a center = (long, lat), and a distance dist in meters, finds the latitudes lat1 and lat2
    such that the locations (x, lat1) and (x, lat2) are dist meters away from the center
    """
    x, y = center
    c = cos(y*pi/180.0)
    d = sin(y*pi/180.0)
    w = 2*(sin(dist/12742000.0))**2
    n1 = -c*(w-1) + d*sqrt(w*(2-w))
    n2 = -c*(w-1) - d*sqrt(w*(2-w))
    lat1 = 180.0*acos(n1)/pi
    if abs(lat_lon_distance(y, x, lat1, x) - dist) > 1:
        lat1 = -lat1
    
    lat2 = 180.0*acos(n2)/pi
    if abs(lat_lon_distance(y, x, lat2, x) - dist) > 1:
        lat2 = -lat2

    if lat1 > lat2:
        lat1, lat2 = lat2, lat1
    return [lat1, lat2]



def get_bbox_by_dist_radius(center, dist, metric = 'geodesic'):
    if metric == 'geodesic':
        lon1, lon2 = get_horizontal_extremes(center, dist)
        lat1, lat2 = get_vertical_extremes(center, dist)
        return (lon1, lat1, lon2, lat2)
    elif metric == 'euclidian':
        return (center[0]-dist, center[1]-dist, center[0]+dist, center[1]+dist)






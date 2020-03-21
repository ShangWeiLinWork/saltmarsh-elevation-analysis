# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 02:04:07 2019

@author: wei
"""
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import geopandas as gpd


def plot_nlargest_poly(polys, n=20):
    '''
    plot the nlargest polygons
    ===
    parameters
    -polys: GesoDataFrame or GeoSeries
    -n: integer
    '''
    
    nlargest_area = polys.area.nlargest(n) 
    fig_nlargest_poly = plt.figure(figsize=(20,20))
    labels = nlargest_area.index.values
    for i in range(0, n):
        ax = fig_nlargest_poly.add_subplot(4, n/4,i+1)
        ax.set_title(labels[i])
        polys.loc[[nlargest_area.index.values[i]],:].plot(ax=ax)
    
    return
        
def value(x):
    '''
    for getting all pixel values using rasterstats
    '''
    return x.compressed()

def annotate_ax(texts, xs, ys, ax, fontsize=10, color='r'):
    '''
    annotate plot given axis
    ===
    parameters
    -texts: list of text to annotate
    -xs: list of x coords for txts, must be same length as txts
    -ys: list of y coords for txts, must be same length as txts
    '''
    for text, x, y in zip(texts, xs, ys):
        ax.annotate(text, (x, y), fontsize=fontsize, color=color)
    
    return

def get_bbox(polys, write=None, prj=None):
    '''
    parameter
    -polys: GeoDataFrame or GeoSeries
    -write: filename. If not given will not save file.
    ===
    return
    -bbox: GeoDataFrame
    '''
    W, S, E, N = polys.total_bounds
    bbox = Polygon([(W, S), (W, N), (E, N), (E, S)])
    bbox = gpd.GeoDataFrame({'geometry':[bbox]})
    
    if prj:
        bbox.crs = prj
    else:
        bbox.crs=polys.crs
        
    if write:
        bbox.to_file(filename = write)
    
    return bbox
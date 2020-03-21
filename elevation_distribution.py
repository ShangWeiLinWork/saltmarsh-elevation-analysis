# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 23:31:49 2019

@author: wei
"""
from rasterstats import zonal_stats
import rasterio.plot
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from other import value
import geopandas as gpd
from other import annotate_ax

def get_downstream_table(stream):
    '''
    parameters
    stream: GeoDataFrame read from Esri hydrolic routine output shp
    '''
    # construct downstream segment relation for each river segment
    # ArcGIS outputs stream in upstream to downstream order
    # so stream B is directly downstream to stream A if first point of B is the last point of A
    downstream_table = stream.copy()
    N_segments = len(stream)
    for i in range(0, N_segments):
        for j in range(i+1, N_segments):
            if downstream_table.loc[i,'TO_NODE'] == downstream_table.loc[j, 'FROM_NODE']:
                downstream_table.loc[i,'downstream_id'] = j
                break

    # get additional distance of each stream segments by recursively adding downstream distance
    downstream_table.loc[:,'length'] = downstream_table.length
    for i in range(0, N_segments):
        distance = 0
        downstream = downstream_table.loc[i,'downstream_id']
        while not np.isnan(downstream):
            distance +=  downstream_table.loc[downstream,'length']
            downstream = downstream_table.loc[downstream, 'downstream_id'] 
        downstream_table.loc[i, 'downstream_distance'] = distance
    return downstream_table

def elevation_distribution(elevation, polys, min_pixel, background, thresholds, stream=None):
    '''
    gets the boxplot of polys in the study site
    ===
    parameters
    -elevation: numpy 2d array
    -polys: GeoSeries or GeoDataFrame
    -background: elevation data, of type rasterio.io.DatasetReader
    -stream: GeoDataFrame read from ArcGIS hydrolic routine output shp.
    '''
    affine = background.transform
    polys_stats = zonal_stats(polys, elevation,
                             affine=affine, nodata=np.nan, stats='count', add_stats={'value':value})
    polys_stats_df = pd.DataFrame(polys_stats, index=polys.index.values)
    
    big_polys_ids = polys_stats_df.loc[:, 'count'] > min_pixel
    big_polys = polys[big_polys_ids]
    big_polys_centroids = big_polys.centroid
    big_polys_stat_df = polys_stats_df[big_polys_ids]
        
    # get down stream table
    downstream_table = get_downstream_table(stream)
        
    # use centroid to find the stream segment the polygon is closest to
    distance_to_stream = np.array([big_polys_centroids.distance(stream_i['geometry']).tolist() for i, stream_i in downstream_table.iterrows()])
    closest_stream_ids = distance_to_stream.argmin(axis=0)

    # get tide travel distance
    # 1. find distance up the segment
    # ArcGIS outputs stream in upstream to downstream order
    # sahpely 'projection' methods calculate distance from first point in LineString
    # so distance up the segment = length_of_segment - distnace_by_projection
    # 2. then add down stream distance
    distance_up_stream = np.empty(len(big_polys))
    for i, big_poly_centroid in enumerate(big_polys_centroids):
        closest_stream = downstream_table.loc[[closest_stream_ids[i]],]
        distance = closest_stream.length - closest_stream.project(big_poly_centroid) + closest_stream.downstream_distance
        distance_up_stream[i] = distance

    big_polys_stat_df.loc[:,'distance_up_stream'] = distance_up_stream

    # boxplot
    fig, ax = plt.subplots(figsize=(30,5))
    ax.boxplot(big_polys_stat_df.loc[:,'value'], positions=big_polys_stat_df.loc[:,'distance_up_stream'],
                  widths=100);
    ax.set_xlabel('distance up stream (m)', fontsize=30)
    ax.set_ylabel('elevation (m)', fontsize=30)
    ax.set_ylim(thresholds[0], thresholds[1])
    ax.tick_params(axis='both', which='major', labelsize=30)

    labels = big_polys_stat_df.index.values
    annotate_ax(labels, big_polys_stat_df.loc[:,'distance_up_stream'], [0.5]*len(big_polys_stat_df), ax, fontsize=30)
    xticks = np.arange(math.floor(min(distance_up_stream)/1000)*1000, math.ceil(max(distance_up_stream)/1000)*1000, 500)
    #min(distance_up_stream) #2929
    #max(distance_up_stream) #18627
    plt.xticks(xticks, xticks)
        
    # plot map
    fig_map, ax_map = plt.subplots(1,1, figsize=(10, 10))
    rasterio.plot.show(background, ax=ax_map, cmap='pink')
    stream.plot(ax=ax_map, color='blue')
    # plot big_pixel polys
    big_polys.plot(ax=ax_map, color='g')
    big_polys_centroids.plot(ax=ax_map, color='r')
    annotate_ax(labels, big_polys_centroids.x, big_polys_centroids.y, ax_map)

    return fig, fig_map

        
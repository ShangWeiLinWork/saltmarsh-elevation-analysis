# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 18:51:13 2019

@author: wei
"""
import numpy as np
import matplotlib.pyplot as plt
import rasterio.mask
import rasterio.plot
import math

def elevation_profile(elevation, nodata, background, p1, p2, interval, ax, style):
    '''
    plot elevation profile
    ===
    parameters
    elevation: numpy 2d array
    nodata: the nodata value in elevation
    background: elevation data, of type rasterio.io.DatasetReader
    p1, p2: start and end coordinates of the elevation profile
    interval: approximate intervals for the elevation profile. the resulting interval will be slightly less than or equal to the given value
    ax: of type matplotlib.axes._subplots.AxesSubplot
    style: string to specify style of elevation profile
    '''
    
    # get sample point coordinates
    distance = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    steps = distance//interval + 2
    xs = np.linspace(p1[0], p2[0], steps)
    ys = np.linspace(p1[1], p2[1], steps)
    
    # get sample rows, cols
    rows_cols = [background.index(x, y) for x, y in (zip(xs, ys))]
    
    # get sample values
    elevations = np.array([elevation[row, col] for row, col in rows_cols])

    # plot
    x_plot = np.array(range(int(steps)))*(distance/(steps-1))
    ax.plot(x_plot, elevations/1000, style)
    ax.set_ylabel('elevation (m)', fontsize=12)
    ax.set_xlabel('distance from p1 (m)', fontsize=12)
             
    return (xs, ys, elevations)

def compare_elevation_profiles(elevation_before, elevation_after, polys,
                              nodata, background, p1, p2,
                              cmap='gray', alpha=0.5, dpi=100):
    '''
    get and overlay two elevation profile
    ===
    parameters
    -elevation_before, elevation_after: elevations to compare. must have same nodata value. numpy 2d array
    -polys: list of GeoSeries or GeoDataFrame. can be list of single or two element.
    -nodata: the nodata value in elevations
    -background: elevation data, of type rasterio.io.DatasetReader
    -p1, p2: start and end coordinates of the elevation profile
    -interval: approximate intervals for the elevation profile. the resulting interval less than or equal to the given value
    -cmap, alpha, dpi: pyplot parameters
    ===
    '''
    pixel_size = background.transform[0]
    fig_profile, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=dpi)   
    
    # plot elevation _profile
    xs, ys, e1 = elevation_profile(elevation_before, nodata, background, p1, p2, pixel_size, ax1, 'bo-')
    _, _, e2 = elevation_profile(elevation_after, nodata, background, p1, p2, pixel_size, ax1, 'ro-')
    
    # plot map
    color = ['y', 'm']
    for i, poly in enumerate(polys):
        poly.plot(ax=ax2, color='None', edgecolor=color[i])
    ax2.plot(xs, ys, 'r')
    difference_id = (~np.isnan(e1)) & (e1!=e2)
    ax2.plot(xs[difference_id], ys[difference_id], 'bX', markersize=2)
    ax2.plot(p1[0], p1[1], 'rX')
    ax2.plot(p2[0], p2[1], 'rX')
    ax2.annotate('p1', (p1[0] + 5, p1[1]), color='r', fontsize=10)
    ax2.annotate('p2', (p2[0] + 5, p2[1]), color='r', fontsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.locator_params(axis='x', nbins=4)
    
    # plot background
    xmin, ymin, xmax, ymax = poly.total_bounds
    ax2.set_xlim([xmin-25,xmax+25])
    ax2.set_ylim([ymin-25,ymax+25])

    rasterio.plot.show(background, cmap=cmap, vmin=0, vmax=5000)
    fig_profile.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9)
    
    
    return fig_profile

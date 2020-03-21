# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 18:51:14 2019

@author: wei
"""
import numpy as np
import rasterio.mask
import rasterio.plot
from scipy import ndimage
from rasterstats import zonal_stats
from shapely.ops import cascaded_union
from elevation_profile import compare_elevation_profiles
from fracture_calculation import fracture_index
import pandas as pd
import matplotlib.pyplot as plt
from elevation_distribution import elevation_distribution
from other import value
import numpy.ma as ma


def polys_filter(elevation, polys, affine, return_raster=False):
    '''
    sets pixel that has centroid outside of polys to nan
    ===
    parameter
    -elevation: 2d numpy array
    -polys: GesoDataFrame or GeoSeries
    -affine: 6-element tuple showing the affine transform of the elevation array
    -return_raster: if true will return a raster that shows which pixels have valid elevation values
    ===
    output
    -poly_elevation: 2d numpy array with values outside of polys set to nan
    -poly_raster: return if return_raster is true. 2d numpy array with boolean values showing which pixels have none-nan values
    '''
    
    # raster
    rasterized_polys = rasterio.features.rasterize(polys.geometry, transform=affine, out_shape=elevation.shape, fill=np.nan)
    
    # elevation
    poly_elevation = np.copy(elevation)
    poly_elevation[np.isnan(rasterized_polys)] = np.nan
    
    if not return_raster:
        return poly_elevation
    
    poly_raster = ~np.isnan(poly_elevation)
    return poly_elevation, poly_raster
    

def threshold_filter(elevation, thresholds, return_raster=False):
    '''
    sets pixel to nan by threshold values
    ===
    parameter
    -elevation: 2d numpy array
    -thresholds: tuple or list specifying lower and upper bounds
    -return_raster: if true will return a raster that shows which pixels have valid elevation values
    ===
    output
    -thresholded_elevation: 2d numpy array with thresholded values set to nan
    -thresholded_raster: return if return_raster is true. 2d numpy array with boolean values showing which pixels have none-nan values
    '''
    
    thresholded_elevation = np.copy(elevation)
    thresholded_elevation[(elevation < thresholds[0]) | (elevation > thresholds[1])] = np.nan
    
    if not return_raster:
        return thresholded_elevation
    
    thresholded_raster = ~np.isnan(thresholded_elevation)
    return thresholded_elevation, thresholded_raster

def difference_filter(elevation, filterable_elevation, polys, affine, pixel_threshold, buffer_range, return_raster=False):
    '''
    find a difference threshold between pixels and their 8-neighborhood with a refernce poly (1std)
    and set pixels over the threshold to nan
    ===
    parameter
    -elevation: 2d numpy array
    -filterable_elevation: 2d numpy array that does not have nan values
    -polys: GesoDataFrame or GeoSeries
    -affine: 6-element tuple showing the affine transform of the elevation array
    -pixel_range: polys having valid pixels less than this value will not be used as reference
    -return_raster: if true will return a raster that shows which pixels have valid elevation values
    ===
    output
    -filtered_elevation: 2d numpy array with values too different from its neighbour set to nan
    -filtered_raster: return if return_raster is true. 2d numpy array with boolean values showing which pixels have none-nan values
    '''
    
    kernel = np.ma.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    filtering_elevation = ndimage.correlate(filterable_elevation, kernel, mode='reflect')
    reference_poly = get_reference_poly(elevation, polys, affine, pixel_threshold)
    eroded_reference_poly = reference_poly.buffer(-buffer_range)
    std = zonal_stats(eroded_reference_poly, filtering_elevation, affine=affine, stats=['std'])[0]['std']
    print(std)
    filtered_elevation = np.copy(elevation)
    filtered_elevation[(filtering_elevation < -std) | (filtering_elevation > std)] = np.nan
    
    if not return_raster:
        return filtered_elevation
    
    filtered_raster = ~np.isnan(filtered_elevation)
    return filtered_elevation, filtered_raster
    
def get_reference_poly(elevation, polys, affine, pixel_threshold):
    '''
    find a reference poly that has largest valid pixel count to circumferenceat ratio among polys that have at least pixel_threshold number of valid pixels
    ===
    parameter
    -elevation: 2d numpy array
    -polys: GesoDataFrame or GeoSeries
    -affine: 6-element tuple showing the affine transform of the elevation array
    -pixel_threshold: lower limit of valid pixel count for a poly to be considered as reference poly
    ===
    return
    -reference_poly: single element GeoSeries

    '''
    
    # filter by valid pixel
    polys_pixel_count = zonal_stats(polys, elevation, affine=affine, stats="count")
    polys_pixel_count = np.array([poly['count'] for poly in polys_pixel_count])
    valid_array = polys_pixel_count > pixel_threshold
    reference_poly = polys.geometry.loc[(polys_pixel_count[valid_array]/polys[valid_array].area).idxmax()]
    
    return reference_poly

def compare_maps(raster_before, raster_after, background, cmap='gray', alpha=0.5, dpi=500, figsize=(10,10), ax=None):
    '''
    plot before and after operation at pixel level
    ===
    parameters
    -raster_before, raster_after: before and after operation results. numpy 2d array
    -background: elevation data. of type rasterio.io.DatasetReader
    -alpha, dpi, figsize: pyplot parameters
    ===
    '''
    if not ax:
        fig, ax = plt.subplots(1,1, figsize=figsize)
        
    difference = np.logical_xor(raster_before, raster_after)
    difference_masked = ma.masked_array(difference, mask=~difference)
    rasterio.plot.show(background, ax=ax, cmap=cmap, vmin=0, vmax=5000)
    rasterio.plot.show(difference_masked, transform=background.transform, ax=ax, alpha=0.5, cmap='autumn')
    
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    return fig

def case_study(elevation_data, polys, thresholds, reference_pixel, sample_poly_id, dist_direction, min_pixel, stream=None, outpath=None):
    '''
    full case study workflow
    ===
    parameters
    -elevation_data: rasterio reader
    -polys: GeoDataFrame or GeoSeries
    -thresholds: tuple of (min_elevation, max_elevation)
    -reference_pixel: lower limit of valid pixel count for a poly to be considered as reference poly (for difference filter step)
    -sample_poly_id: the id of the sample polygon from polys
    -dist_direction: direction for plotting elevaiton distriution. currently has two options: 'WE' plots distance in west-east direction; 'upstream' plots distance upstream
    -min_pixel: lower limit of valid pixel count for a poly to be considered in the elevation distribution process
    -stream: required if direction is 'upstream', GeoDataFrame read from ArcGIS hydrolic routine output shp.
    '''
    affine = elevation_data.transform
    pixel_size = affine[0]
    elevation = elevation_data.read(1)
    
# =============================================================================
#     # plot n largest shape
#     nlargest_area = polys.loc[:,'area'].nlargest(20) 
#     fig_nlargest_poly = plt.figure(figsize=(20,20))
#     for i in range(0, 20):
#         ax = fig_nlargest_poly.add_subplot(4, 20/4,i+1)
#         polys.loc[[nlargest_area.index.values[i]],:].plot(ax=ax)
# =============================================================================
    
    # poly
    poly_elevation, poly_raster = polys_filter(elevation, polys, affine, return_raster=True)
    
    # threshold
    thresholded_elevation, thresholded_raster = threshold_filter(poly_elevation, thresholds, return_raster=True)
    
    # erode
    eroded_polys = polys.buffer(-pixel_size)
    eroded_polys = eroded_polys.loc[~eroded_polys.is_empty]
    eroded_elevation, eroded_raster = polys_filter(thresholded_elevation, eroded_polys, affine, return_raster=True)
    
    # difference filter
    filterable_elevation = np.copy(elevation)
    filterable_elevation[(elevation < thresholds[0])] = thresholds[0]
    filterable_elevation[(elevation > thresholds[1])] = thresholds[1]
    fig,ax=plt.subplots(1, 1)
    filtered_elevation, filtered_raster = difference_filter(eroded_elevation, filterable_elevation, polys, affine, reference_pixel, pixel_size, return_raster=True)
    
    # statistics
    union_poly = cascaded_union(polys.geometry)
    union_eroded_poly = cascaded_union(eroded_polys)
    stats = ['mean', 'std', 'median', 'percentile_25', 'percentile_75','count']
    poly_stat = zonal_stats(union_poly, poly_elevation,
                            affine=affine, nodata=np.nan, stats=stats, add_stats={'value':value})
    thresholded_stat = zonal_stats(union_poly, thresholded_elevation,
                            affine=affine, nodata=np.nan, stats=stats, add_stats={'value':value})
    eroded_stat = zonal_stats(union_eroded_poly, eroded_elevation,
                            affine=affine, nodata=np.nan, stats=stats, add_stats={'value':value})
    filtered_stat = zonal_stats(union_eroded_poly, filtered_elevation,
                            affine=affine, nodata=np.nan, stats=stats, add_stats={'value':value})
    stat_df = pd.DataFrame(poly_stat + thresholded_stat + eroded_stat + filtered_stat,
                           index=['raw', 'threshold', 'erode', 'filter']).drop(columns=['value'])
    stat_df = stat_df.round(3)
    print(stat_df)
    
    # elevation profile
    sample_poly = polys.loc[[sample_poly_id],]
    eroded_sample_poly = eroded_polys.loc[[sample_poly_id],]
    xmin, ymin, xmax, ymax = sample_poly.total_bounds
    x = (xmin + xmax) / 2
    p1 = (x, ymin)
    p2 = (x, ymax)
    fig_elev_threshold = compare_elevation_profiles(poly_elevation, thresholded_elevation, [sample_poly], np.nan, elevation_data, p1, p2)
    fig_elev_threshold.suptitle('threshold by HAT, MSL')
    fig_elev_threshold.savefig(outpath+'threshold_elevation_profile.png')
    plt.close(fig_elev_threshold)
    fig_elev_erode = compare_elevation_profiles(thresholded_elevation, eroded_elevation, [sample_poly, eroded_sample_poly], np.nan, elevation_data, p1, p2)
    fig_elev_erode.suptitle('inward buffer {} m'.format(pixel_size))
    fig_elev_erode.savefig(outpath+'erode_elevation_profile.png')
    plt.close(fig_elev_erode)
    fig_elev_filter = compare_elevation_profiles(eroded_elevation, filtered_elevation, [sample_poly, eroded_sample_poly], np.nan, elevation_data, p1, p2)
    fig_elev_filter.suptitle('filter by difference')
    fig_elev_filter.savefig(outpath+'filter_elevation_profile.png')
    plt.close(fig_elev_filter)
    fig_elev_total = compare_elevation_profiles(poly_elevation, filtered_elevation, [sample_poly, eroded_sample_poly], np.nan, elevation_data, p1, p2)   
    fig_elev_total.suptitle('all operations')
    fig_elev_total.savefig(outpath+'total_elevation_profile.png')
    plt.close(fig_elev_total)
     
#     # fracture index
#     frac_index, fig_frac = fracture_index(polys, pixel_size, 1, pixel_size, False)
#     fig_frac.suptitle('fracture_index')
#     
#     # maps
    fig_threshold = compare_maps(poly_raster, thresholded_raster, elevation_data)
    fig_threshold.suptitle('thresholded')
    fig_threshold.savefig(outpath+'threshold_map.png')
    plt.close(fig_threshold)
    fig_erode = compare_maps(thresholded_raster, eroded_raster, elevation_data)
    fig_erode.suptitle('inward buffer {} m'.format(pixel_size))
    fig_erode.savefig(outpath+'erode_map.png')
    plt.close(fig_erode)
    fig_filter = compare_maps(eroded_raster, filtered_raster, elevation_data)
    fig_filter.suptitle('filter by difference')
    fig_filter.savefig(outpath+'filter_map.png')
    plt.close(fig_filter)
    fig_total = compare_maps(poly_raster, filtered_raster, elevation_data)
    fig_total.suptitle('all operations')
    fig_total.savefig(outpath+'total_map.png')
    plt.close(fig_total)
     
    fig_dist_hist, fig_dist_map = elevation_distribution(filtered_elevation, polys, dist_direction, min_pixel, elevation_data, thresholds, stream)
    fig_dist_hist.savefig(outpath+'dist_histogram.png')
    fig_dist_map.savefig(outpath+'dist_map.png')
    plt.close(fig_dist_map)
# =============================================================================

    return {'poly_elevation': poly_elevation, 'thresholded_elevation': thresholded_elevation, 'eroded_elevation': eroded_elevation, 'filtered_elevation': filtered_elevation,
           'stat_df': stat_df, 'eroded_polys': eroded_polys}
# =============================================================================
#    return {'poly_elevation': poly_elevation, 'thresholded_elevation': thresholded_elevation, 'eroded_elevation': eroded_elevation, 'filtered_elevation': filtered_elevation,
#             'stat_df': stat_df, 'eroded_polys': eroded_polys, 'fractrue_index': frac_index}
# 
# =============================================================================

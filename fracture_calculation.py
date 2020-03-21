# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 18:47:55 2019

@author: wei
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio.plot


def downscale_raster_by_sum(raster, ratio):
    '''
    downscale raster to lower resolution by summing pixel values
    ===
    parameters
    raster: single band, numpy array
    ratio: downscaling ratio, must be an integer and common factor of the number of rows and columns of the raster
    '''
    # to blocks
    raster_shape = raster.shape
    raster_reshaped = raster.reshape(int(raster_shape[0]/ratio), ratio, int(raster_shape[1]/ratio), ratio)
    # sum pixel in each block
    #raster_reshaped_sum = np.nansum(raster_reshaped, axis=(1,3))
    raster_reshaped_sum = raster_reshaped.sum(axis=(1, 3))
    return raster_reshaped_sum

def dissolve_polys_gdf(polys_gdf):
    '''
    dissolve geopandas GeoDataFrame to single polygon
    ===
    parameters
    polys_gdf: GeoDataFrame
    ===
    return
    GeoSeries
    '''
    polys_gdf['dissolve'] = 1
    return polys_gdf.dissolve(by='dissolve', aggfunc = 'first')['geometry']

def fracture_index(polys, pixel_size, calculation_pixel_size, buffer_range, plot_process):
    '''
    plots the fracture index of the site
    ===
    parameter
    -polys: GeoDataFrame or GeoSeries
    -pixel_size: output pixel size, each pixel will have a fracture_index value
    -calculation_pixel_size: pixel used in calculation. calculation pixel is smaller than pixel, and is used to get the coverage information of polys within each pixel
    -buffer_range: the buffer range for the outward and inward buffering procedure for getting the generalized area
    -plot_process: if true, will plot process
    ===
    return
    -fig_result: map of fracture index, matplotlib figure
    -fig, fig2, fig3: plots of process, matplotlib figure
    '''
    
    # buffer
    buffered_polys = gpd.GeoDataFrame()
    buffered_polys['geometry'] = polys.buffer(buffer_range)
    # dissolve
    dissolved_buffered_poly = dissolve_polys_gdf(buffered_polys)
    # erode
    eroded_dissolved_buffered_poly = dissolved_buffered_poly.buffer(-buffer_range)

    # get affine transforms
    # for result
    bounds = polys.total_bounds
    N = ((bounds[3] // pixel_size) + 2) * pixel_size
    S = (bounds[1] // pixel_size - 2) * pixel_size
    E = ((bounds[2] // pixel_size) + 2) * pixel_size
    W = (bounds[0] // pixel_size - 2) * pixel_size
    ncol = int((E - W) / pixel_size)
    nrow = int((N - S) / pixel_size)
    affine = rasterio.transform.from_bounds(W, S, E, N, ncol, nrow)
    # for calculation
    ratio = int(pixel_size/calculation_pixel_size)
    ncol2 = ncol*ratio 
    nrow2 = nrow*ratio
    affine2 = rasterio.transform.from_bounds(W, S, E, N, ncol2, nrow2)

    # rasterize
    polys_raster = rasterio.features.rasterize(polys['geometry'], transform=affine2, out_shape=(nrow2, ncol2))
    generalized_raster = rasterio.features.rasterize(eroded_dissolved_buffered_poly, transform=affine2, out_shape=(nrow2, ncol2))
    nerror = sum(sum(polys_raster>generalized_raster))
    print('{} pixels are occupied by polys_raster but not generalized_raster, they will be forced to 1'.format(nerror))
    generalized_raster[polys_raster == 1] = 1
    
    # downscale
    downscaled_polys_raster = downscale_raster_by_sum(polys_raster, ratio)
    downscaled_generalized_raster = downscale_raster_by_sum(generalized_raster, ratio)
    
    # fracture_index
    fracture_raster = downscaled_polys_raster/downscaled_generalized_raster
    
    
    # plot process
    if plot_process:
        # fig1: vector processing
        fig, ax = plt.subplots(1, 4, figsize=(10, 10))
        polys['id'] = polys.index.values
        polys.plot(ax=ax[0], column='id')
        # buffer
        ax[1].set_title('buffer')
        buffered_polys['id'] = polys.index.values
        buffered_polys.plot(ax=ax[1], column='id')
        # dissolve
        ax[2].set_title('dissolve')        
        dissolved_buffered_poly.plot(ax=ax[2])
        # erode
        ax[3].set_title('erode')
        eroded_dissolved_buffered_poly.plot(ax=ax[3])
        polys.plot(ax=ax[3], alpha=0.5, color='r')
        
        # fig2: rasterize
        fig2, ax = plt.subplots(1, 2)
        extent = [W, E, S, N]
        polys.plot(ax=ax[0], zorder=0)
        ax[0].imshow(polys_raster, cmap='Greys', extent=extent, alpha=0.5, zorder=1)
        eroded_dissolved_buffered_poly.plot(ax=ax[1], color='r', zorder=0)
        ax[1].imshow(generalized_raster, cmap='Greys', extent=extent, alpha=0.5, zorder=1)
        
        # fig3: downscale
        fig3, ax = plt.subplots(1, 2)
        ax[0].imshow(downscaled_generalized_raster, origin='upper', extent=extent)
        ax[1].imshow(downscaled_polys_raster, origin='upper', extent=extent)
        
    extent = [W, E, S, N]
    fig_result, ax_result = plt.subplots(1, 1, figsize=(10, 10))
    fracture_plot = ax_result.imshow(fracture_raster, cmap='cool_r')
    fig_result.colorbar(fracture_plot, ax=ax_result)
    
    if not plot_process:
        return (fracture_raster, fig_result)
    
    return (fracture_raster, fig_result, fig, fig2, fig3)


# test_polys = polys_data.loc[[1457, 1459],:]
#fracture_index(test_polys, 5, 1, 5, True, True)
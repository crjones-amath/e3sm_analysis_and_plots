#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of plotting routines / utilities for E3SM. This should be organized
later, but want to get this down to clean up my scripts.

Created on Wed Nov 21 12:28:15 2018

@author: christopher.jones@pnnl.gov
"""

# imports
import numpy as np
# import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs   # map plots
import cartopy.feature as cfeature   # needed for map features
import cartopy.crs as ccrs   # map plots
import cartopy.feature as cfeature   # needed for map features



#####################################
# Unstructured grid utilities
#####################################

# entries are (corner_max, edge_max)
gll_area_max_thresholds = {'ne30': (0.0001, 0.00025),
                           'ne120': (0.000006, 0.000015)}


def classify_gll_nodes(ds, res='ne30', add_mask_to_coords=True):
    """ Classify unstructured E3SM grid nodes based on area
    """
    max_corner, max_edge = gll_area_max_thresholds[res]
    area = ds.area.values  # convert to a numpy array
    corner = area <= max_corner
    edge = np.logical_and(area > max_corner, area <= max_edge)
    center = area > max_edge

    mask = 1 * corner + 2 * edge + 3 * center
    if add_mask_to_coords:
        ds.coords['mask'] = ('ncol', mask)
        ds.mask.attrs['description'] = 'mask identify gll node classification'
        ds.mask.attrs['long_name'] = 'gll_node_type'
        ds.mask.attrs['units'] = ' '
        ds.mask.attrs['flags'] = '1: corner node, 2: edge node, 3: center node'
    return mask


def awm(da, area, region):
    """return area-weighted mean of dataset over region

    Note: redundant with area_weighted_mean; need to find a better way to
    deal with (da, area) and (ds, variable) ways of specifying this."""
    if region is None:
        return (da * area).sum(dim='ncol') / area.sum(dim='ncol')
    else:
        num = (da * area).where(region).sum(dim='ncol')
        denom = area.where(region).sum(dim='ncol')
        return num / denom


def plot_profile_by_area(ds, variable, area, region, mask,
                         mask_vals=[1, 2, 3],
                         labels=['corner', 'edge', 'center'],
                         do_pert=True, figsize=(12, 6)):
    """plot profiles over region separated by corner/edge/center"""
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    da = ds[variable]
    if do_pert:
        # first calculate anomaly
        da = da - awm(da, area, region).mean(dim='time')
    awm(da, area, region).mean(dim='time').plot(ax=ax, linestyle='--',
                                                color='k', y='lev',
                                                label='all')
    for val, lab in zip(mask_vals, labels):
        reg = np.logical_and(region, mask == val)
        awm(da, area, reg).mean(dim='time').plot(ax=ax, y='lev', label=lab)
    ax.legend()
    ax.invert_yaxis()
    ax.set_xlabel(variable + ' (' + ds[variable].units + ')')
    return ax


def area_weighted_mean(da=None, area=None, ds=None, var=None):
    """area-weighted mean of dataArray da given an 'area' dataArray
    """
    if ds is not None and var is not None:
        return area_weighted_mean(da=ds[var], area=ds['area'])
    dsum = [d for d in da.dims if d != 'time']
    return (da * area).sum(dim=dsum) / area.sum(dim=dsum)


def area_weighted_rmse(da, area):
    """area-weighted RMSE of anomaly da"""
    return np.sqrt(area_weighted_mean(da ** 2, area))


def toa_title(da, area, model_name, show_mean=True,
              show_rmse=False, units="", fmt='{:.2g}'):
    """Calculate mean and/or RMSE and return string to use as title in plots

    inputs:
        da - xarray dataarray with the data for spatial means
        model_name - model name to include in string
        show_mean - add global area-weighted mean to output string if true
        show_rmse - add global area-weighted rmse to output string if true
        units - optionally specify units
        fmt - optionally specify formatting string form mean and rmse
    output: string of form "(mean) (model_name) (rmse) (units)"
    """
    if show_mean:
        mn = ("Mean: " + fmt).format(area_weighted_mean(da, area))
    else:
        mn = ""
    if show_rmse:
        rmse = ("RMSE: " + fmt).format(area_weighted_rmse(da, area))
    else:
        rmse = ""
    return "{:12} {} {:>12}  ".format(mn, model_name, rmse) + units


# Create a feature for states
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')


def map_layout(ax, *features, extent=None):
    """Add cartopy map to axis ax.

    sample features:
        cfeature.LAKES, cfeature.RIVERS
    """
    ax.add_feature(cfeature.LAND, alpha=0.2)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=0.5)
    ax.add_feature(states_provinces, edgecolor='black', alpha=0.2)
    for feat in features:
        ax.add_feature(feat)
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        ax.set_global()


def plot_global(v, ds, time_slice=None, projection=ccrs.LambertCylindrical(),
                extent=None, rescale=1, units="", name='SP2-ECPP',
                cmap=None, mask_threshold=None, ilev=None, **kwargs):
    """ 2D Map plot of time-mean of ds[v].

    Arguments:
        v - variable to plot
        ds - xarray dataset containing variable v
        time_slice - optionally specify slice for time-mean.
                     if None, do mean over all times.
        projection - cartopy projection to use for map
        extent - optionally specify region for plotting (global plot if None)
        rescale - factor to rescale output by, so plotted_v = v * rescale
        units - optional string to specify units on map titles
        name - name used in plot title
        cmap - optionally specify colormap
        mask_threshold - optionally mask out values below mask_threshold
    """
    if time_slice is None and 'time' in ds:
        da = ds[v].mean(dim='time') * rescale
    elif 'time' in ds:
        da = ds[v].sel(time=time_slice).mean(dim='time') * rescale
    else:
        da = ds[v] * rescale
    if ilev is not None:
        da = da.isel(lev=ilev)

    # set title before masking because my global mean functions are dumb
    ax_title = toa_title(da, ds['area'], model_name=name,
                         show_mean=True, show_rmse=False, units=units)
    if mask_threshold is not None:
        da.values[da.values < mask_threshold] = np.nan
    fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': projection})

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    if cmap is None:
        p = da.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cbar_ax=cax,
                             robust=True)
        # fig.colorbar(p, orientation='horizontal')
    else:
        da.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cbar_ax=cax,
                         robust=True, cmap=cmap)        
    map_layout(ax, extent=extent)
    ax.set_title(ax_title)
    
def multi_model_canvas(n, figsize=None):
    """subplot array (nrows, ncols) = (n, n+1) with last column scaled 5% for colorbar
    
    Helper function called by multi_model_global_plot.
    """
    if figsize is None:
        figsize = (8*n, 3*n)
    width_ratios = [1] * n + [0.05]
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n, n + 1, width_ratios=width_ratios)
    return fig, gs


def multi_model_global_plot(v, ds_dict, names, time_slice=None, isel_times=None,
                            projection=ccrs.LambertCylindrical(), extent=None, 
                            rescale=1, units="", cmap_div=plt.cm.RdBu_r, fmt='{:.3g}',
                            vmin=None, vmax=None, cmap=None):
    """Plot time-mean of variable v (and anomalies) on global map for datasets in ds_dict
    
    Each column corresponds to a model run. The top row plots the time-mean of the data.
    The second row shows the anomaly of "v" relative to the data in the first column,
    the 3rd row shows the anomaly of "v" relative to the data in the second column, 
    and so on. Colorbars at the end of each row are common for the row.
    
    Inputs:
        v - variable to plot
        ds_dict - dictionary with (key, val) = (model_name, dataset)
        names - list containing the names of models to show on plot
                (expected to be keys of ds_dict)
        time_slice - optionally specify slice for time-mean.
                     if None, do mean over all times.
        isel_times - optionally specify indices for time-mean
                     only valid if time_slice is None
        projection - cartopy projection to use for map
        extent - optionally specify region for plotting (if None, do global plot)
        rescale - factor to rescale output by, so plotted_v = v * rescale
        units - optional string to specify units on map titles
        cmap_div - optionally specify colormap for the "delta" plots
        fmt - formatting string passed to toa_title
        vmin, vmax - optionally specify range for colorbar
        
    Notes:
      - names is emptied while this function is called
      - I need to find a better way to set mins and maxes of the colorbars, but I want 
        to make sure that the colorbar matches the colormaps for each subplot. I should
        see if this approach here is really necessary.
    """
    
    if cmap is None:
        cmap = plt.cm.viridis  # default colormap
    fig, gs = multi_model_canvas(len(names))
    
    # prep data_array
    dat = {key: ds_dict[key][v] * rescale for key in names if key in ds_dict}
    
    # try to get units if none are supplied
    if not units:
        for key in dat:
            try:
                units = ds_dict[key][v].attrs['units']
            except:
                pass
    
    # subset in time and average
    for key, val in dat.items():
        if 'time' not in [d for d in val.dims]:
            continue
        if time_slice is None:
            if isel_times is not None:
                dat[key] = val.isel(time=isel_times).mean(dim="time")
            else:
                dat[key] = val.mean(dim="time")
        else:
            dat[key] = val.sel(time=time_slice).mean(dim="time")
            
    # top row: just plot the data
    if vmin is None:
        vmin = np.min([np.nanmin(v.values) for v in dat.values()])
    if vmax is None:
        vmax = np.max([np.nanmax(v.values) for v in dat.values()])
    
    cbax = fig.add_subplot(gs[0, -1])
    for col, model in enumerate(names):
        ax = fig.add_subplot(gs[0, col], projection=projection)
        if model in dat:
            dat[model].plot.contourf(ax=ax, transform=ccrs.PlateCarree(),
                                     cbar_ax=cbax, vmin=vmin, vmax=vmax, cmap=cmap)
            map_layout(ax, extent)
            ax.set_title(toa_title(dat[model], model_name=model, show_mean=True, show_rmse=False, units=units, fmt=fmt))
    
    # for remaining rows, plot differences:
    # note to self: someday find a better way to do this
    row = 0
    key = names.pop(0) # pop first element -- this will be the new "obs"
    while names:
        row = row + 1
        d0 = dat.pop(key)

        # update data with difference from d0
        dat.update((k, val - d0) for k, val in dat.items())
        vmin = np.min([np.nanmin(v.values) for v in dat.values()])
        vmax = np.max([np.nanmax(v.values) for v in dat.values()])
        
        vmax = np.max([abs(vmin), abs(vmax)])
        vmin = -vmax
        cbax = fig.add_subplot(gs[row, -1])
        
        for col, model in enumerate(names):
            ax = fig.add_subplot(gs[row, row + col], projection=projection)
            if model in dat:
                dat[model].plot.contourf(ax=ax, transform=ccrs.PlateCarree(),
                                         cbar_ax=cbax, vmin=vmin, vmax=vmax, cmap=cmap_div, robust=True)
                map_layout(ax, extent)
                ax_name = ' - '.join([model, key])
                ax.set_title(toa_title(dat[model], model_name=ax_name, show_mean=True, show_rmse=True, units=units, fmt=fmt))
        key = names.pop(0)
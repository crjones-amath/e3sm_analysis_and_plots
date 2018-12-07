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
import matplotlib.pyplot as plt
import cartopy.crs as ccrs   # map plots
from .map_utils import map_layout, map_axes_with_vertical_cb 


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
    return fig, ax


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
        mean_val = area_weighted_mean(da, area).values.item()
        mn = ("Mean: " + fmt).format(mean_val)
    else:
        mn = ""
    if show_rmse:
        rmse_val = area_weighted_rmse(da, area).values.item()
        rmse = ("RMSE: " + fmt).format(rmse_val)
    else:
        rmse = ""
    return "{:12} {} {:>12}  ".format(mn, model_name, rmse) + units


def plot_global(v, ds, time_slice=None, projection=ccrs.PlateCarree(),
                extent=None, rescale=1, units="", name='SP-ECPP',
                mask_threshold=None, ilev=None, figsize=(8, 6),
                **kwargs):
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
    lat = ds['lat'].values
    lon = ds['lon'].values
    lon[lon > 180] = lon[lon > 180] - 360  # convert to -180, 180
    # set title before masking because my global mean functions are dumb
    ax_title = toa_title(da, ds['area'], model_name=name,
                         show_mean=True, show_rmse=False, units=units)
    if mask_threshold is not None:
        da.values[da.values < mask_threshold] = np.nan
    fig, ax, cax = map_axes_with_vertical_cb(figsize=figsize,
                                             projection=projection)
    p = ax.tripcolor(lon, lat, da, transform=ccrs.PlateCarree(), **kwargs)
    cb = plt.colorbar(p, cax=cax, label=v + " (" + units + ")")
    map_layout(ax, extent=extent)
    ax.set_title(ax_title)
    return fig, ax, cax, cb

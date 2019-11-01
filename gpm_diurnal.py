import xarray as xr
import pandas as pd

if __name__ == "__main__":
    topdir = '/global/cscratch1/sd/crjones/obs/gpm/'
    yrs = ['2014', '2015', '2017', '2018', '2016']
    diurnal_cycle = {}
    for yr in yrs:
        print(yr)
        try:
            ds = xr.open_mfdataset(topdir + yr + '/3B*.nc')
            diurnal_cycle[yr] = ds['precipitationCal'].groupby('time.hour').mean(dim='time').load()
        except:
            print('something failed for year ' + yr)
    
    # save outputs
    keys = sorted([key for key in diurnal_cycle.keys()])
    dsout = xr.concat([diurnal_cycle[k] for k in keys], dim=pd.Index(keys, name='year'))
    dsout.to_netcdf(topdir + 'gpm-MAM-diurnal.nc')

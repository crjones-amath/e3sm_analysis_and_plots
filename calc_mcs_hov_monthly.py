"""
This script calculates monthly mean MCS precipitation map and Hovmoller diagram using pixel-level MCS tracking files
and saves the monthly output to two separate netCDF files.
To run this code:
python calc_mpas_mcs_monthly_precipmaphov_single.py year month runname
For example:
python calc_mpas_mcs_monthly_precipmaphov_single.py 2010 5 0.25deg_MCS-EUS
"""
import numpy as np
import glob, sys, os
import xarray as xr
import time, datetime, calendar, pytz

__author__ = "Zhe Feng, zhe.feng@pnnl.gov"
__copyright__ = "Zhe Feng, zhe.feng@pnnl.gov"

# Get input from command line
year = (sys.argv[1])
month = (sys.argv[2]).zfill(2)
runname = (sys.argv[3])
#exp = sys.argv[4]
#year = '2001'
#month = '05'
#runname = '0.5deg_MCS-EUS'
exp = ''

rootdir = f'/global/cscratch1/sd/feng045/FACETS/mpas_production/{runname}/'
mcsdir = f'{rootdir}{year}/mcstracking_testno10mm/'
outdir = f'{rootdir}{year}/stats{exp}_testno10mm/'
#mcsdir = f'{rootdir}{year}/mcstracking/'
#outdir = f'{rootdir}{year}/stats{exp}/'
mcsfiles = sorted(glob.glob(f'{mcsdir}mcstrack_{year}{month}??_????.nc'))
print(mcsdir)
print(year, month)
print('Number of files: ', len(mcsfiles))

map_outfile = f'{outdir}mcs_rainmap_{year}{month}.nc'
hov_outfile = f'{outdir}mcs_rainhov_{year}{month}.nc'
os.makedirs(outdir, exist_ok=True)

# Hovmoller domain
startlat = 31.0
endlat = 48.0
startlon = -110.0
endlon = -80.0


# Read data
ds = xr.open_mfdataset(mcsfiles, concat_dim='time', drop_variables=['numclouds','tb','cloudnumber'])
print('Finish reading input files.')
# ds.load()

ntimes = ds.dims['time']

# Create an array to store number of MCSs on a map
nmcs_map = np.full((len(ds.lat), len(ds.lon)), 0)

# Find the min/max track number
mintracknum = ds.pcptracknumber.min().values.item()
maxtracknum = ds.pcptracknumber.max().values.item()
# Some months has 0 MCS, in which case a NAN is returned
# In that case set the min/max track number to 0
if np.isnan(mintracknum) == True:
    mintracknum = 0
    maxtracknum = 0
else:
    mintracknum = int(mintracknum)
    maxtracknum = int(maxtracknum)

# Loop over each track number
for itrack in range(mintracknum, maxtracknum+1):
#     print(itrack)
    # Locate the same track number across time, sum over time to get the swath,
    # turn the swath to 1 (true/false), then sum on the map
    nmcs_map += ((ds.pcptracknumber.where(ds.pcptracknumber == itrack).sum(dim='time')) > 0).values


# Sum MCS precipitation over time, use cloudtracknumber > 0 as mask
mcsprecip = ds['precipitation'].where(ds.cloudtracknumber > 0).sum(dim='time')

# Sum total precipitation over time
totprecip = ds['precipitation'].sum(dim='time')


# Convert all MCS track number to 1 for summation purpose
mcspcpmask = ds.pcptracknumber.values
mcspcpmask[mcspcpmask > 0] = 1

# Convert numpy array to DataArray
mcspcpmask = xr.DataArray(mcspcpmask, coords={'time':ds.time, 'lat':ds.lat, 'lon':ds.lon}, dims=['time','lat','lon'])

# Sum MCS PF counts overtime to get number of hours
mcspcpct = mcspcpmask.sum(dim='time')

# Compute Epoch Time for the month
months = np.zeros(1, dtype=int)
months[0] = calendar.timegm(datetime.datetime(int(year), int(month), 1, 0, 0, 0, tzinfo=pytz.UTC).timetuple())

# Define xarray dataset for Map
dsmap = xr.Dataset({'precipitation': (['time', 'lat', 'lon'], totprecip.expand_dims('time', axis=0)), \
                    'mcs_precipitation': (['time', 'lat', 'lon'], mcsprecip.expand_dims('time', axis=0)), \
                    'mcs_precipitation_count': (['time', 'lat', 'lon'], mcspcpct.expand_dims('time', axis=0)), \
                    'mcs_number':(['time', 'lat', 'lon'], np.expand_dims(nmcs_map, axis=0)), \
                    'ntimes': (['time'], xr.DataArray(ntimes).expand_dims('time', axis=0))}, \
                    coords={'time': (['time'], months), \
                            'lat': (['lat'], ds.lat), \
                            'lon': (['lon'], ds.lon)}, \
                    attrs={'title': 'MCS precipitation accumulation', \
                           'contact':'Zhe Feng, zhe.feng@pnnl.gov', \
                           'created_on':time.ctime(time.time())})

dsmap.time.attrs['long_name'] = 'Epoch Time (since 1970-01-01T00:00:00)'
dsmap.time.attrs['units'] = 'Seconds since 1970-1-1 0:00:00 0:00'

dsmap.lon.attrs['long_name'] = 'Longitude'
dsmap.lon.attrs['units'] = 'degree'

dsmap.lat.attrs['long_name'] = 'Latitude'
dsmap.lat.attrs['units'] = 'degree'

dsmap.ntimes.attrs['long_name'] = 'Number of hours in the month'
dsmap.ntimes.attrs['units'] = 'count'

dsmap.precipitation.attrs['long_name'] = 'Total precipitation'
dsmap.precipitation.attrs['units'] = 'mm'

dsmap.mcs_precipitation.attrs['long_name'] = 'MCS precipitation'
dsmap.mcs_precipitation.attrs['units'] = 'mm'

dsmap.mcs_precipitation_count.attrs['long_name'] = 'Number of hours MCS precipitation is recorded'
dsmap.mcs_precipitation_count.attrs['units'] = 'hour'

dsmap.mcs_number.attrs['long_name'] = 'Number of individual MCS passing a grid point'
dsmap.mcs_number.attrs['units'] = 'count'

fillvalue = np.nan
dsmap.to_netcdf(path=map_outfile, mode='w', format='NETCDF4_CLASSIC', unlimited_dims='time', \
                encoding={'lon':{'zlib':True, 'dtype':'float32'}, \
                          'lat':{'zlib':True, 'dtype':'float32'}, \
                          'precipitation':{'zlib':True, '_FillValue':fillvalue, 'dtype':'float32'}, \
                          'mcs_precipitation':{'zlib':True, '_FillValue':fillvalue, 'dtype':'float32'}, \
                          'mcs_precipitation_count':{'zlib':True, '_FillValue':fillvalue, 'dtype':'float32'}, \
                          'mcs_number':{'zlib':True, '_FillValue':fillvalue, 'dtype':'float32'}, \
                          })
print('Map output saved as: ', map_outfile)



#####################################################################
# This section calculates Hovmoller
#####################################################################
# Mask out non-MCS precipitation, assign to a tmp array
mcspreciptmp = ds['precipitation'].where((ds.pcptracknumber > 0) & (ds.precipitation >= 0)).values
# Replace NAN with 0, for averaging Hovmoller purpose
mcspreciptmp[np.isnan(mcspreciptmp)] = 0

# Convert numpy array to DataArray
mcspcp = xr.DataArray(mcspreciptmp, coords={'time':ds.time, 'lat':ds.lat, 'lon':ds.lon}, dims=['time','lat','lon'])

# Select a latitude band and time period where both simulation exist
mcspreciphov = mcspcp.sel(lat=slice(startlat, endlat)).mean(dim='lat')
totpreciphov = ds['precipitation'].where(ds.precipitation >= 0).sel(lat=slice(startlat, endlat)).mean(dim='lat')

# Convert xarray decoded time back to Epoch Time in seconds
basetime = np.array([tt.tolist()/1e9 for tt in ds.time.values])

# Define xarray dataset for Hovmoller
print('Writing Hovmoller to netCDF file ...')
dshov = xr.Dataset({'precipitation': (['time', 'lon'], totpreciphov), \
                    'mcs_precipitation': (['time', 'lon'], mcspreciphov), \
                    }, \
                    coords={'lon': (['lon'], ds.lon), \
                            'time': (['time'], basetime)}, \
                    attrs={'title': 'MCS precipitation Hovmoller', \
                           'startlat':startlat, \
                           'endlat':endlat, \
                           'startlon':startlon, \
                           'endlon':endlon, \
                           'contact':'Zhe Feng, zhe.feng@pnnl.gov', \
                           'created_on':time.ctime(time.time())})

dshov.lon.attrs['long_name'] = 'Longitude'
dshov.lon.attrs['units'] = 'degree'

dshov.time.attrs['long_name'] = 'Epoch Time (since 1970-01-01T00:00:00)'
dshov.time.attrs['units'] = 'seconds since 1970-01-01T00:00:00'

dshov.precipitation.attrs['long_name'] = 'Total precipitation'
dshov.precipitation.attrs['units'] = 'mm/h'

dshov.mcs_precipitation.attrs['long_name'] = 'MCS precipitation from regridding (Control)'
dshov.mcs_precipitation.attrs['units'] = 'mm/h'

dshov.to_netcdf(path=hov_outfile, mode='w', format='NETCDF4_CLASSIC', unlimited_dims='time', \
                encoding={'time': {'zlib':True, 'dtype':'int32'}, \
                          'lon':{'zlib':True, 'dtype':'float32'}, \
                          'precipitation':{'zlib':True, 'dtype':'float32'}, \
                          'mcs_precipitation':{'zlib':True, 'dtype':'float32'}, \
                         })
print('Hovmoller output saved as: ', hov_outfile)


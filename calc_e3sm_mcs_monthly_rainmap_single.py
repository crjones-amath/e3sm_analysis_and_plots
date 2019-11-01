import numpy as np
import glob, sys, os
import xarray as xr
import time, datetime, calendar, pytz
from concurrent.futures import ProcessPoolExecutor
from itertools import product

# sdate = sys.argv[1]
# edate = sys.argv[2]
# year = (sys.argv[3])
# month = (sys.argv[4]).zfill(2)
#sdate = '20010301'
#edate = '20011031'
#year = '2001'
#month = '05'

def calc_rainmap(args):
    top_subdir = args[0]
    year = top_subdir[0:4]
    month = args[1].zfill(2)

    mcsdir = f'/global/cscratch1/sd/crjones/ECP/e3sm/mcstracking/{top_subdir}/'
    outdir = '/global/cscratch1/sd/crjones/ECP/e3sm/statstb/monthly/'
    mcsfiles = sorted(glob.glob(mcsdir + 'mcstrack_' + year + month + '??_????.nc'))
    print(mcsdir)
    print(year, month)
    print('Number of files: ', len(mcsfiles))

    map_outfile = outdir + 'mcs_rainmap_' + year + month + '.nc'
    os.makedirs(outdir, exist_ok=True)


    # Read data
    ds = xr.open_mfdataset(mcsfiles, concat_dim='time', drop_variables=['numclouds','tb','cloudnumber'])
    print('Finish reading input files.')
    # ds.load()


    # Create an array to store number of MCSs on a map
    nmcs_map = np.full((len(ds.lat), len(ds.lon)), 0)

    # Find the min/max track number
    mintracknum = int(ds.pcptracknumber.min().values.item())
    maxtracknum = int(ds.pcptracknumber.max().values.item())

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

    ntimes = ds.dims['time']


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
    
def main():
    top_subdirs = ['20020301_20021031', '20030301_20031031']
    # start_dates = ['20020301', '20030301']
    # end_dates = ['20021031', '20031031']
    months = [str(m).zfill(2) for m in range(3, 11)]
    with ProcessPoolExecutor(max_workers=4) as Executor:
        Executor.map(calc_rainmap, product(top_subdirs, months))
    # for args in product(top_subdirs, months):
    #     print(args)
        
    

if __name__ == "__main__":
    main()
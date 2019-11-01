"""Split a series of netcdf"""
import xarray as xr
# from dask_jobqueue import SLURMCluster
# from dask.distributed import Client, LocalCluster, progress
import glob
import os
from concurrent.futures import ProcessPoolExecutor

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'


# years = ['0003', '0004', '0005', '0006', '0007']
# start_dates = ['0003-12-01', None, None, None, None]

years = ['0002']
start_dates = ['0002-04-01']

# MMF
# path_out = '/global/project/projectdirs/m3312/crjones/e3sm/early_science/hourly_2d_hist/remap/daily/'
# out_name = path_out + 'earlyscience.FC5AV1C-H01A.ne120.sp1_64x1_1000m.cam.h1.'
# path_in = '/global/project/projectdirs/m3312/crjones/e3sm/early_science/hourly_2d_hist/remap/'

# E3SM
path_out = '/global/project/projectdirs/m3312/crjones/e3sm/early_science_e3sm/hourly_2d_hist/remap/daily/'
out_name = path_out + 'earlyscience.FC5AV1C-H01A.ne120.E3SM.cam.h1.'
path_in = '/global/project/projectdirs/m3312/crjones/e3sm/early_science_e3sm/hourly_2d_hist/remap/'

files_in = sorted(glob.glob(path_in + '*.nc'))
dates_in = [f.split('.')[-2][:10] for f in files_in]  # selects yyyy-mm-dd from $case.yyyy-mm-dd-sssss.nc
years_in = {d[0:4] for d in dates_in}
files_by_year = {y: [f for f in files_in if y in f] for y in years_in}
dates_by_year = {y: [d for d in dates_in if y in d] for y in years_in}

def file_list_for_given_year(year, start_date=None):
    # may need to grab one of the previous year as well, since 000{n}-12-31 will run over into 000{n+1}
    file_list = files_by_year[year]
    start_idx = dates_by_year[year].index(start_date) if start_date in dates_by_year[year] else 0
    if start_idx > 0:
        return file_list[start_idx:]
    else:
        # need to include last element of previous year
        prior_year = str(int(year) - 1).zfill(4)
        prepend = [files_by_year[prior_year][-1]] if prior_year in years_in else []
        return prepend + file_list

def main():    
    # loop over years, split into daily files
    for yr, start_date in zip(years, start_dates):
        print('Processing year ', yr)
        files_to_check = file_list_for_given_year(yr, start_date=start_date)
        ds = xr.open_mfdataset(files_to_check, parallel=True).sel(time=yr).chunk(chunks={'time': 12})
        days, dsets = zip(*ds.groupby('time.dayofyear'))
        out_files = [out_name + dsets[d].time[0].item().strftime('%Y-%m-%d').replace(' ', '0') + '.nc' for d in range(len(days))]
        previously_processed = glob.glob(path_out + '*.nc')
        out_to_do = [out_files[d] for d in range(len(days)) if out_files[d] not in previously_processed]
        dsets_to_do = [dsets[d] for d in range(len(days)) if out_files[d] not in previously_processed]
        if out_to_do:
            print('First file: ' + out_to_do[0])
            print('Last file: ' + out_to_do[-1])
            xr.save_mfdataset(dsets_to_do, out_to_do)
            

def split_file_to_daily(fname):
    ds = xr.open_dataset(fname)
    days, dsets = zip(*ds.groupby('time.dayofyear'))
    out_files = [out_name + dsets[d].time[0].item().strftime('%Y-%m-%d').replace(' ', '0') + '.nc' for d in range(len(days))]
    previously_processed = glob.glob(path_out + '*.nc')
    out_to_do = [out_files[d] for d in range(len(days)) if out_files[d] not in previously_processed]
    dsets_to_do = [dsets[d] for d in range(len(days)) if out_files[d] not in previously_processed]
    if out_to_do:
        print('First file: ' + out_to_do[0])
        print('Last file: ' + out_to_do[-1])
        xr.save_mfdataset(dsets_to_do, out_to_do)

def main_alt(do_parallel=True, max_workers=8):
    """Previous version dies a lot, so try alternate approach ..."""
    for yr, start_date in zip(years, start_dates):
        print('Processing year ', yr)
        files_to_check = file_list_for_given_year(yr, start_date=start_date)
        if do_parallel:
            with ProcessPoolExecutor(max_workers=max_workers) as Executor:
                Executor.map(split_file_to_daily, files_to_check)
        else:  # only for testing
            for fname in files_to_check[:1]:
                split_file_to_daily(fname)

if __name__ == "__main__":
    # cluster = LocalCluster(n_workers=6)
    # client = Client()

    # main()
    
    # this version is much faster when it is known that each file 
    # contains full days (i.e., no day's output is split across multiple files)
    main_alt(do_parallel=True)

    
"""Remap a bunch of files in parallel"""
from concurrent.futures import ProcessPoolExecutor
import glob
import subprocess

# files_sp = glob.glob('/global/project/projectdirs/m3312/crjones/e3sm/early_science/hourly_2d_hist/*.000[3-7]-*.nc')
# outdir = '/global/project/projectdirs/m3312/crjones/e3sm/early_science/hourly_2d_hist/remap/'

resolution = 'ne30'
case = 'SP'

mapfiles = {'ne30': '/global/homes/z/zender/data/maps/map_ne30np4_to_cmip6_180x360_aave.20181001.nc',
            'ne120': '/global/homes/z/zender/data/maps/map_ne120np4_to_cmip6_720x1440_aave.20181001.nc'}

# case 1: E3SM, res = ne30
if case == 'E3SM':
    if resolution == 'ne30':
        files = sorted(glob.glob('/global/project/projectdirs/m3312/whannah/earlyscience.FC5AV1C-L.ne30.E3SM.20190519/atm/*.cam.h1.000?-0[2-8]*.nc'))
        outdir = '/global/project/projectdirs/m3312/crjones/e3sm/earlyscience.FC5AV1C-L.ne30.E3SM.20190519/remap/'
    elif resolution == 'ne120':
        files = sorted(glob.glob('/global/project/projectdirs/m3312/crjones/e3sm/early_science_e3sm/hourly_2d_hist/*.000[2-3]-*.nc'))
        outdir = '/global/project/projectdirs/m3312/crjones/e3sm/early_science_e3sm/hourly_2d_hist/remap/'
elif case == 'SP':
    if resolution == 'ne30':
        files = sorted(glob.glob('/global/project/projectdirs/m3312/crjones/e3sm/earlyscience.FC5AV1C-L.ne30.sp1_64x1_1000m.20190415/hourly_2d_hist/*.cam.h1.000[0-5]-0[2-8]*.nc'))
        outdir = '/global/project/projectdirs/m3312/crjones/e3sm/earlyscience.FC5AV1C-L.ne30.sp1_64x1_1000m.20190415/hourly_2d_hist/remap/'

mapfile = mapfiles[resolution]

def remap_file(fname, skip_processed=True):
    print('processing file ' + fname)
    result = subprocess.run(['ncremap', '-m', mapfile, '-a', 'conserve', '-O', outdir, fname])
    if result.returncode != 0:
        print('Remap failed for fname: ' + fname)
        print('Return code: ' + str(result.returncode))
    return result.returncode

def main(do_parallel=True):
    if not do_parallel:
        remap_file(files[0])  # test
    else:
        # remap the files
        with ProcessPoolExecutor(max_workers=8) as Executor:
            Executor.map(remap_file, files)

if __name__ == "__main__":
    main()

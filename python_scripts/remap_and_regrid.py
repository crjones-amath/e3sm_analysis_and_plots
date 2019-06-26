"""Remap a bunch of files in parallel"""
from concurrent.futures import ProcessPoolExecutor
import glob
import subprocess

# files_sp = glob.glob('/global/project/projectdirs/m3312/crjones/e3sm/early_science/hourly_2d_hist/*.000[3-7]-*.nc')
# outdir = '/global/project/projectdirs/m3312/crjones/e3sm/early_science/hourly_2d_hist/remap/'

files_sp = sorted(glob.glob('/global/project/projectdirs/m3312/crjones/e3sm/early_science_e3sm/hourly_2d_hist/*.000[2-3]-*.nc'))
outdir = '/global/project/projectdirs/m3312/crjones/e3sm/early_science_e3sm/hourly_2d_hist/remap/'

mapfile = '/global/homes/z/zender/data/maps/map_ne120np4_to_cmip6_720x1440_aave.20181001.nc'

def remap_file(fname, skip_processed=True):
    print('processing file ' + fname)
    result = subprocess.run(['ncremap', '-m', mapfile, '-a', 'conserve', '-O', outdir, fname])
    if result.returncode != 0:
        print('Remap failed for fname: ' + fname)
        print('Return code: ' + str(result.returncode))
    return result.returncode

def main(do_parallel=True):
    if not do_parallel:
        remap_file(files_sp[0])  # test
    else:
        # remap the files
        with ProcessPoolExecutor(max_workers=4) as Executor:
            Executor.map(remap_file, files_sp)

if __name__ == "__main__":
    main()

#!/usr/bin/env qmap
#workload_manager=local
#conda=boonza
#concurrency=16

import boonza
import pandas as pd
import os

! rm -rf md_final
! mkdir -p md_final

df = pd.read_csv('analysis.csv').sort_values(by='drmsd').reset_index(drop=True)
for idx, row in df.iterrows():
    drmsd  = row['drmsd']
    folder = row['folder']
    name   = f'{idx:03d}_{drmsd:.3f}_{folder}'
    ifile  = row['workdir'] + '/final.mae'

    if os.path.exists(ifile):
        ! boonza convert ${ifile} md_final/${name}.mae -s "not water and not ions and not name Na Cl NA CL SOD CLA" --structure-only


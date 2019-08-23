#!/bin/bash
#PBS -N lanzi
#PBS -q stat
#PBS -l nodes=1:ppn=6:m128G,walltime=168:00:00
#PBS -V
##PBS -m bae -M shiwei@illinois.edu

cd ~/STGP/code/AD
module load matlab/9.5

# install mtimesx in matlab on linux cluster
# blas_lib='/usr/local/matlab/R2018b/bin/glnxa64/libmwblas.so'
# mex('-DDEFINEUNIX','-largeArrayDims','mtimesx.c',blas_lib)

# matlab -nodesktop -nodisplay < plot_extdcorpoi_f.m > plot_extdcorpoi_f.out &
mcc -m plot_extdcorpoi_f.m -I ~/STGP/code/AD -I ~/STGP/code/util -I ~/STGP/code/util/STGP/ -I ~/STGP/code/util/ndSparse/ -I ~/STGP/code/util/Image_Graphs/ -I ~/STGP/code/util/tight_subplot/;
chmod +x ./run_plot_extdcorpoi_f.sh
./run_plot_extdcorpoi_f.sh /usr/local/matlab/R2018b $grp_opt $mdl_opt [] [] $intM

# ./plot_extdcorpoi_f
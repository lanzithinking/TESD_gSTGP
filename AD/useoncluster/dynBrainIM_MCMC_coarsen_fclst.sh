#!/bin/bash
#PBS -N lanzi
#PBS -q stat
#PBS -l nodes=1:ppn=6:m128G,walltime=100:00:00
#PBS -V
##PBS -m bae -M shiwei@illinois.edu

cd ~/STGP/code/AD
module load matlab/9.5

# install mtimesx in matlab on linux cluster
# blas_lib='/usr/local/matlab/R2018b/bin/glnxa64/libmwblas.so'
# mex('-DDEFINEUNIX','-largeArrayDims','mtimesx.c',blas_lib)

# matlab -nodesktop -nodisplay < dynBrainIM_MCMC_coarsen_fclst.m > dynBrainIM_MCMC_coarsen_fclst.out &
mcc -m dynBrainIM_MCMC_coarsen_fclst.m -I ~/STGP/code/AD -I ~/STGP/code/util -I ~/STGP/code/sampler -I ~/STGP/code/util/STGP_mtimesx/ -I ~/STGP/code/util/mtimesx/ -I ~/STGP/code/util/Image_Graphs/;
chmod +x ./run_dynBrainIM_MCMC_coarsen_fclst.sh
./run_dynBrainIM_MCMC_coarsen_fclst.sh /usr/local/matlab/R2018b $grp_opt $mdl_opt [] [] [] $intM

# ./dynBrainIM_MCMC_coarsen_fclst
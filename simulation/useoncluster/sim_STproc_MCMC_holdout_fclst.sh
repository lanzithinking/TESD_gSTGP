#!/bin/bash
#PBS -N lanzi
#PBS -q stat
#PBS -l nodes=1:ppn=6:m128G,walltime=100:00:00
#PBS -V
##PBS -m bae -M shiwei@illinois.edu

cd ~/STGP/code/simulation
module load matlab/9.5

# install mtimesx in matlab on linux cluster
# blas_lib='/usr/local/matlab/R2018b/bin/glnxa64/libmwblas.so'
# mex('-DDEFINEUNIX','-largeArrayDims','mtimesx.c',blas_lib)

# matlab -nodesktop -nodisplay < sim_STproc_MCMC_holdout_fclst.m > sim_STproc_MCMC_holdout_fclst.out &
mcc -m sim_STproc_MCMC_holdout_fclst.m -I ~/STGP/code/simulation -I ~/STGP/code/util -I ~/STGP/code/sampler;
chmod +x ./run_sim_STproc_MCMC_holdout_fclst.sh
./run_sim_STproc_MCMC_holdout_fclst.sh /usr/local/matlab/R2018b $mdl_opt $num_trl

# ./sim_STproc_MCMC_holdout_fclst
# submit:
# qsub -v mdl_opt=2,num_trl=100 -N simhld2K100 -q stat sim_STproc_MCMC_holdout_fclst.sh
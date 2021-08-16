#!/bin/bash
 
#SBATCH -N 1                        # number of compute nodes
#SBATCH -n 1                        # number of cores
#SBATCH --mem=16G                   # GigaBytes of memory required (per node)

#SBATCH -p gpu                      # Use gpu partition
#SBATCH -q wildfire                 # Run job under wildfire QOS queue
#SBATCH --gres=gpu:1                # Request two GPUs

#SBATCH -t 0-12:00                  # wall time (D-HH:MM)
##SBATCH -A slan7                   # Account hours will be pulled from (commented out with double # in front)
#SBATCH -o %x.log                   # STDOUT (%j = JobId)
#SBATCH -e %x.err                   # STDERR (%j = JobId)
#SBATCH --mail-type=END,FAIL             # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=slan@asu.edu    # send-to address

# load environment
module load matlab/2019b
MCRROOT=/packages/7x/matlab/2019b
export LD_LIBRARY_PATH=${MCRROOT}/bin/glnxa64:$LD_LIBRARY_PATH

# go to working directory
cd ~/Projects/STGP/code/simulation

# install mtimesx in matlab on linux cluster
# blas_lib='/packages/7x/matlab/2019b/bin/glnxa64/libmwblas.so'
# mex('-DDEFINEUNIX','-largeArrayDims','mtimesx.c',blas_lib)

# matlab -nodesktop -nodisplay < simSTproc_MCMC_fclst.m > simSTproc_MCMC_fclst.out &
mcc -m simSTproc_MCMC_fclst.m -I ~/Projects/STGP/code/simulation -I ~/Projects/STGP/code/sampler -I ~/Projects/STGP/code/util -I ~/Projects/STGP/code/util/STGP_gpu;
chmod +x ./run_simSTproc_MCMC_fclst.sh
./run_simSTproc_MCMC_fclst.sh ${MCRROOT} $mdl_opt $num_trl [] [] [] $intM $hold_out

# ./simSTproc_MCMC_fclst
# submit:
# sbatch --export=mdl_opt=2,num_trl=100,intM=1,hold_out=0 --job-name=sim2K100 --output=sim2K100.out simSTproc_MCMC_fclst.sh
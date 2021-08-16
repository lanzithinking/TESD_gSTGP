#!/bin/bash
 
#SBATCH -N 1                        # number of compute nodes
#SBATCH -c 4                        # number of "tasks" (cores)
#SBATCH --mem=64G                   # GigaBytes of memory required (per node)

#SBATCH -p serial                   # partition 
#SBATCH -q normal                   # QOS

#SBATCH -t 2-00:00                  # wall time (D-HH:MM)
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
mcc -m simSTproc_MCMC_fclst.m -I ~/Projects/STGP/code/simulation -I ~/Projects/STGP/code/sampler -I ~/Projects/STGP/code/util -I ~/Projects/STGP/code/util/mtimesx -I ~/Projects/STGP/code/util/STGP_mtimesx;
chmod +x ./run_simSTproc_MCMC_fclst.sh

for num_trl in 100 1000
do
	for mdl_opt in {0..2}
	do
		echo -e " Seed Number $seedNO: model $mdl_opt with $num_trl trials...\n "
		./run_simSTproc_MCMC_fclst.sh ${MCRROOT} $num_trl $stationary $mdl_opt $upthypr [] $intM $hold_out $spdapx $seedNO
	done
done

# ./simSTproc_MCMC_fclst
# submit:
# sbatch --export=stationary=0,upthypr=1,intM=0,hold_out=1,spdapx=0,seedNO=2019 --job-name=hld-2019 --output=hld-2019.log simSTproc_MCMC_fclst_hldout.sh
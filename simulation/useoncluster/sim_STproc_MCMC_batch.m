% batch job

job=batch('sim_STproc_MCMC');

wait(job);
diary(job,'sim_STproc_MCMC_diary');
load(job);

delete(job);
clear job;
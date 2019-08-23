% batch job
function []=sim_STproc_MCMC_f_batch(mdl_opt)
job=batch('sim_STproc_MCMC_f',0,{mdl_opt});

wait(job);
diary(job,['sim_STproc_MCMC_f_diary',num2str(mdl_opt)]);
load(job);

delete(job);
clear job;
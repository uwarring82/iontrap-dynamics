svals=[];
ETHvals=[];
varvals=[];
vsta = -0.2;
vend = 2.7;
vstep = (vend-vsta)/50;
%eners = [1.,1.73,2.41,3.05,3.67];
%optfactor = mean(eners(1:N));
omegaAx = 0.707;
OmegaR = 1.28; 
nval = [1,1,1];
N=3
%for i=1:1:N
%nval = [nval;[0.5]];
%end
for cutoff = 1:1:7;
sval = []; 
varval = [];
ETHval = [];
export = [];
tsta = 0;
tend = 14/OmegaR;
tend = tend * 2 * pi
tstep = (tend-tsta)/100;
for omegaz = vsta:vstep:vend
[P,nph,meansz,meansx,meansy,varsz,varsx,varsy] = sb_evolution_modes_Bloch(N,1,cutoff,nval,OmegaR,omegaz,omegaAx,0,0,tsta,tend,tstep);
[meanE,DE,enerf,szdiag,sxdiag,sz,sx,sy,ener] = ergodic(N,cutoff,nval,OmegaR,omegaz,omegaAx,0,0);
sval   = [sval;[omegaz,meansz,meansx,meansy]];
varval = [varval;[omegaz,varsz,varsx,varsy]];
ETHval = [ETHval;[omegaz,sz,sx,sy]];
export = [export;[omegaz, meansz, varsz, sz, N, cutoff, OmegaR, omegaAx, tend]];
disp(['omegaz = ',num2str(omegaz)])
end
export=table(export);
filename=['/home/gclos/Dokumente/QSIM/Paula/SpinBoson/' num2str(N) 'ions_matlabtheory_nbar1_cutoff' num2str(cutoff) '.txt']
writetable(export,filename,'Delimiter','tab'),
svals = [svals,[sval]];
ETHvals = [ETHvals,[ETHval]];
varvals = [varvals,[varval]];
end
save('nions4_allinone_exp_theory_nbar01_cutoffs.mat')
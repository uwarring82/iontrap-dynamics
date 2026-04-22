svals=[];
ETHvals=[];
varvals=[];
realrabis = [0.73,0.95,1.28,1.37];
realwax = [0.724,0.707,0.707,0.708]
%realtemps = [[0.75 0 0 0], [0.25 0.7 0 0] , [0.7 1.0 1.0 0], [1.0 1.5 1.5 1.5]];
realtemps = [[0.5 0.5 0.5 0.5], [0.5 0.5 0.5 0.5] , [0.5 0.5 0.5 0.5], [0.5 0.5 0.5 0.5]];
ncuts = [6,5,4,3]
vsta = -0.3;
vend = 3;
vstep = (vend-vsta)/50;
%eners = [1.,1.73,2.41,3.05,3.67];
%optfactor = mean(eners(1:N));
%omegaAx = 0.707;
%OmegaR = optfactor * omegaAx;
%nval=[];
%for i=1:1:N
%nval = [nval;[0.5]];
%end
for N = 1:1:4;
sval = []; 
varval = [];
ETHval = [];
export = [];
omegaAx = realwax(N);
OmegaR = realrabis(N);
nval = realtemps(4*(N-1)+1:1:4*N);
cutoff = ncuts(N)
tsta = 0;
tend = 14/OmegaR;
tend = tend * 2 * pi
tstep = (tend-tsta)/200;
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
filename=['/home/gclos/Dokumente/QSIM/Paula/SpinBoson/' num2str(N) 'ions_matlabtheory_nbar05.txt']
writetable(export,filename,'Delimiter','tab'),
svals = [svals,[sval]];
ETHvals = [ETHvals,[ETHval]];
varvals = [varvals,[varval]];
end
save('nions4_allinone_exp_theory_nbar05.mat')
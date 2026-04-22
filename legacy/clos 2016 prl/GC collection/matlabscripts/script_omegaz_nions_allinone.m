svals=[];
ETHvals=[];
varvals=[];
for N = 1:1:3;
eners = [1.,1.73,2.41,3.05,3.67];
optfactor = mean(eners(1:N));
sval = []; 
varval = [];
ETHval = [];
omegaAx = 0.707;
OmegaR = optfactor * omegaAx;
vsta = 0;
vend = 2.5;
vstep = (vend-vsta)/80;
tsta = 0;
tend = 10/OmegaR;
tend = tend * 2 * pi
tstep = (tend-tsta)/200;
nval=[];
for i=1:1:N
nval = [nval;[0.5]];
end
for omegaz = vsta:vstep:vend
[P,nph,meansz,meansx,meansy,varsz,varsx,varsy] = sb_evolution_modes_Bloch(N,1,3,nval,OmegaR,omegaz,omegaAx,0,0,tsta,tend,tstep);
[meanE,DE,enerf,szdiag,sxdiag,sz,sx,sy,ener] = ergodic(N,3,nval,OmegaR,omegaz,omegaAx,0,0);
sval   = [sval;[omegaz,meansz,meansx,meansy]];
varval = [varval;[omegaz,varsz,varsx,varsy]];
ETHval = [ETHval;[omegaz,sz,sx,sy]];
disp(['omegaz = ',num2str(omegaz)])
end
svals = [svals,[sval]];
ETHvals = [ETHvals,[ETHval]];
varvals = [varvals,[varval]];
end
save('nions4_allinone_nbar05_t10omega_opt.mat')
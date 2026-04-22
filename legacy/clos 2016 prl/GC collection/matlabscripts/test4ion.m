N=4  
eners = [1.,1.73,2.41];
optfactor = mean(eners(1:3));
sval = [];
varval = [];
ETHval = [];
nval=[];
for i=1:1:N
nval = [nval;[0.5]];
end
for omegaAx = 1:1:1
OmegaR = optfactor * omegaAx;
omegaz = optfactor * omegaAx;
[P,nph,meansz,meansx,meansy,varsz,varsx,varsy] = sb_evolution_modes_Bloch(N,1,3,nval,OmegaR,omegaz,omegaAx,0,0,0.0,200,1);
[meanE,DE,enerf,szdiag,sxdiag,sz,sx,sy,ener] = ergodic(N,3,nval,OmegaR,omegaz,omegaAx,0,0);
sval   = [sval;[omegaAx,meansz,meansx,meansy]];
varval = [varval;[omegaAx,varsz,varsx,varsy]];
ETHval = [ETHval;[omegaAx,sz,sx,sy]];
disp(['omegaz = ',num2str(omegaAx)])
end
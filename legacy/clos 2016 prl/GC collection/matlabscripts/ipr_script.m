



for N = 1:2;
out = []; 
export = [];
omegaAx = 0.7;
vsta = 0 * omegaAx;
vend = 2.5 * omegaAx;
vstep = (vend-vsta)/ 50;
eners = [1.,1.73,2.41,3.05,3.67];
optfactor = mean(eners(1:N));
OmegaR = optfactor*omegaAx; 
nval = [1,1,1,1,1];
cutoff = 5;
tsta = 0;
tend = 14/OmegaR;
tend = tend * 2 * pi
tstep = (tend-tsta)/150;
for omegaz = vsta:vstep:vend
[P,nph,meansz,meansx,meansy,varsz,varsx,varsy] = sb_evolution_modes_Bloch(N,1,cutoff,nval,OmegaR,omegaz,omegaAx,0,0,tsta,tend,tstep);
[meanE,DE,enerf,szdiag,sxdiag,nphdiag_cell,sz,sx,sy,IPR,IPR0,ratio_IPR] = ergodic(N,cutoff,nval,OmegaR,omegaz,omegaAx,0,0)
%[meanE,DE,enerf,szdiag,sxdiag,sz,sx,sy,ener] = ergodic(N,cutoff,nval,OmegaR,omegaz,omegaAx,0,0);
out   = [out;[omegaz/omegaAx,IPR,IPR0,ratio_IPR,meansz,varsz]];
disp(['omegaz = ',num2str(omegaz)])
end
export=table(out);
filename=['/home/gclos/Dokumente/QSIM/Paula/SpinBoson/' num2str(N) 'ions_ipr_test.txt']
writetable(export,filename,'Delimiter','tab'),
figure()
plot(out(:,4),out(:,6))
hold on
end
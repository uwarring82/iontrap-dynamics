for N = 5:5;
out = []; 
export = [];
omegaAx = 0.7;
vsta = -1.1.* omegaAx;
vend = 4. * omegaAx;
vstep = (vend-vsta)/300;
eners = [1.,1.73,2.41,3.05,3.67];
optfactor = mean(eners(1:N));
OmegaR = 0.05*optfactor*omegaAx;%0.02* 
nval = [1,1,1,1,1];
cutoff = 3;
tsta = 0;
tend = 10/OmegaR;
tend = tend * 2 * pi
tstep = (tend-tsta)/50;
for omegaz = vsta:vstep:vend
[P,nph,meansz,meansx,meansy,varsz,varsx,varsy] = sb_evolution_modes_Bloch(N,1,cutoff,nval,OmegaR,omegaz,omegaAx,0,0,tsta,tend,tstep);
%[meanE,DE,enerf,szdiag,sxdiag,sz,sx,sy,ener] = ergodic(N,cutoff,nval,OmegaR,omegaz,omegaAx,0,0);
out   = [out;[omegaz/omegaAx,min(P(2:4:end))]];
disp(['omegaz = ',num2str(omegaz)])
end
export=table(out);
filename=['/home/gclos/Dokumente/QSIM/Paula/SpinBoson/' num2str(N) 'ions_spectrum_weak_min_test.txt']
writetable(export,filename,'Delimiter','tab'),
figure()
plot(out(:,1),out(:,2))
hold on
end
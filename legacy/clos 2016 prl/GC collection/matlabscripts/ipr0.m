cutoff=6;
out = [];
for N = 1:4;
omegaAx = 0.7;
eners = [1.,1.73,2.41,3.05,3.67];
optfactor = mean(eners(1:N));
OmegaR = optfactor*omegaAx;
omegaz = optfactor*omegaAx;
nval = [1,1,1,1,1];
[meanE,DE,enerf,szdiag,sxdiag,nphdiag_cell,sz,sx,sy,IPR,IPR0,ratio_IPR] = ergodic_ipr(N,cutoff,nval,OmegaR,omegaz,omegaAx,0,0);
out   = [out;[N, IPR,IPR0]];
disp(['N, IPR= ',num2str(N), '  ',num2str(IPR),'  ',num2str(IPR0)])
end
export=table(out);
filename=['/home/gclos/Dokumente/QSIM/Paula/SpinBoson/' 'ions_ipr_ipr0.txt']
writetable(export,filename,'Delimiter','tab'),
figure()
hold on
plot(out(:,1),outN(:,2))
plot(out(:,1),outN(:,3))
legend('ipr','ipr0')

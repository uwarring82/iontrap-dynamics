figure()
for cutoff=8;
out = [];
N = 1:3;
for nt = [0.01:0.01:0.1,0.2:0.1:1,1.5:0.5:5,6:1:10,20:10:100,200:100:1000];
omegaAx = 0.7;
eners = [1.,1.73,2.41,3.05,3.67];
optfactor = mean(eners(1:N));
OmegaR = optfactor*omegaAx;
omegaz = optfactor*omegaAx;
nval = nt * [1,1,1,1,1];
[meanE,DE,enerf,szdiag,sxdiag,nphdiag_cell,sz,sx,sy,IPR,IPR0,ratio_IPR] = ergodic_ipr(N,cutoff,nval,OmegaR,omegaz,omegaAx,0,0);
out   = [out;[nt, IPR,IPR0,ratio_IPR,meanE,DE]];
disp(['nt, me, de= ',num2str(nt), '  ',num2str(meanE),'  ',num2str(DE)])
end
%export=table(out);
%filename=['/home/gclos/Dokumente/QSIM/Paula/SpinBoson/' 'ions_Delta_ipr_nc8.txt']
%writetable(export,filename,'Delimiter','tab'),
%figure()
hold on
%plot(log10(out(:,1)),out(:,2),'o')
%plot(log10(out(:,1)),out(:,3),'o')
%figure()
%plot(out(:,1),out(:,4),'o')
%figure()
%plot(log10(out(:,1)),out(:,2)-out(:,3),'o')
plot(out(:,1),out(:,5),'o')
plot(out(:,1),out(:,6),'o')
legend('me','de')
end
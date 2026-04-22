%timings: nc=7,17pts, 90s, nc=8:255s, diff below 2%; more pts:434s,
%nc=9:880s (more pts)
tic;
%omegaAxes = [0.724, 0.707,0.707,0.708,0.709];
%cutoffs = [10,10,10,7,3];
nval = [1,1,1,1,1];
N=3;
eners = [1.,1.73,2.41,3.05,3.67];
optfactor = mean(eners(1:N));
%rabis = [0.73,0.95,1.28,1.37,1.58];
for cutoff=8;
out = [];
for omegaAx=[0.2:0.01:0.39,0.4:0.05:1,1.2:0.2:1.8,2:0.5:5,6:1:10,20:5:50,60:20:100,200:100:500,1000];
OmegaR = optfactor*omegaAx;
omegaz = 0.6;%optfactor*omegaAx;
[meanE,DE,enerf,szdiag,sxdiag,nphdiag_cell,sz,sx,sy,IPR,IPR_av,eta0] = ergodic_ipr_av_new(N,cutoff,nval,OmegaR,omegaz,omegaAx,0,0);
out   = [out;[eta0, IPR_av, IPR, omegaz, OmegaR, N, cutoff]];
disp(['N, wz, Iav, I = ',num2str(N), '  ',num2str(eta0),'  ',num2str(IPR_av),'  ',num2str(IPR)])
end
export=table(round(out,4));
filename=['/home/gclos/Dokumente/QSIM/Paula/SpinBoson/sbpaper/data/' '3ions_IPR_av_eta_max_nc' num2str(cutoff) '.txt']
writetable(export,filename,'Delimiter','tab'),
end
%figure()
%hold on
%plot(log10(out(:,1)),out(:,2),'o')
%plot(log10(out(:,1)),out(:,3),'o')
%figure()
%plot(out(:,1),out(:,4),'o')
%figure()
%plot(log10(out(:,1)),out(:,2)-out(:,3),'o')
%plot(out(:,1),out(:,5),'o')
%plot(out(:,1),out(:,2),'o')
%legend('me','de')
toc;
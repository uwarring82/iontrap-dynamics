%timings: n=4,nc=(5,6,7), 1pt: (71,353,1755)s, ipr@wz=1(13.7,16.1,18.0)
%N=4;30pts,nc=7,53000s
%timings: n=3,nc=(5,6,7,8,9,10,11), 1pt: (0.78,1.66,4.29,11.4,26.1,57.3,116.6)s, ipr@wz=1(8.1,8.5,8.9,9.2,9.3,9.4,9.5)
%timings: n=5,nc=(2,3,4), 1pt: (1.1,36)s, ipr@wz=1(5.6,10.2)
tic;
omegaAxes = [0.724, 0.707,0.707,0.708,0.709];
cutoffs = [10,10,10,7,3];
nval = [1,1,1,1,1];
%eners = [1.,1.73,2.41,3.05,3.67];
rabis = [0.73,0.95,1.28,1.37,1.58];
for N = 4;
out = [];
OmegaR = rabis(N);
omegaAx = omegaAxes(N);
cutoff = cutoffs(N);
for omegaz = 0:0.1:3;
[meanE,DE,enerf,szdiag,sxdiag,nphdiag_cell,sz,sx,sy,IPR,IPR_av,eta0] = ergodic_ipr_av_new(N,cutoff,nval,OmegaR,omegaz,omegaAx,0,0);
out   = [out;[omegaz, IPR_av, IPR, meanE, DE]];
disp(['N, wz, Iav, I = ',num2str(N), '  ',num2str(omegaz),'  ',num2str(IPR_av),'  ',num2str(IPR)])
end
export=table(round(out,4));
filename=['/home/gclos/Dokumente/QSIM/Paula/SpinBoson/' num2str(N) 'ions_IPR_av.txt']
writetable(export,filename,'Delimiter','tab'),
figure()
%hold on
%plot(log10(out(:,1)),out(:,2),'o')
%plot(log10(out(:,1)),out(:,3),'o')
%figure()
%plot(out(:,1),out(:,4),'o')
%figure()
%plot(log10(out(:,1)),out(:,2)-out(:,3),'o')
%plot(out(:,1),out(:,5),'o')
plot(out(:,1),out(:,2),'o')
%legend('me','de')
end
toc;
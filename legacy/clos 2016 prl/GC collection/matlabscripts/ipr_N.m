%timings: nc=6,4pts, 390s
%N=5, nc=(2,3,4) wz=0.6, IPR=(6.6,12.4,19.7) t=(0.5,16.5,?)
tic;
%omegaAxes = [0.724, 0.707,0.707,0.708,0.709];
cutoffs = [10,10,10,7,5];
omegaAx=0.7;
nval = [1,1,1,1,1];
eners = [1.,1.73,2.41,3.05,3.67];
%rabis = [0.73,0.95,1.28,1.37,1.58];
omegazs= [0,0.3,0.6,0.6,0.6];
%cutoff=6;
for diffcutoff=1;%[-3,-2,-1];
out = [];
for N=5;
optfactor = mean(eners(1:N));
OmegaR = optfactor*omegaAx;
omegaz = 0.6;%optfactor*omegaAx;
cutoff=cutoffs(N);
[meanE,DE,enerf,szdiag,sxdiag,nphdiag_cell,sz,sx,sy,IPR,IPR_av,eta0] = ergodic_ipr_av_new(N,cutoff,nval,OmegaR,omegaz,omegaAx,0,0);
out   = [out;[N, IPR_av, IPR, omegaz, OmegaR, eta0, cutoff]];
disp(['N, wz, Iav, I = ',num2str(N), '  ',num2str(eta0),'  ',num2str(IPR_av),'  ',num2str(IPR)])
end
%export=table(round(out,4));
%filename=['/home/gclos/Dokumente/QSIM/Paula/SpinBoson/sbpaper/data/' 'IPR_av_N_eta1_nc' num2str(cutoff) '.txt']
%writetable(export,filename,'Delimiter','tab'),
end
toc;
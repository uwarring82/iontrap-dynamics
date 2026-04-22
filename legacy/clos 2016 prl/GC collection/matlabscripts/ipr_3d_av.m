tic;
figure()
cutoff=6;
N=3;
nval = [1,1,1,1,1];
out = [];
%for omegaAx=[0.2,0.25,0.3:0.1:1,1.5,2:2:10,100,1000]
omegaAx=0.7;
for OmegaR=-0.05:0.01:2.85;
for omegaz=-0.05:0.01:2.85;
[meanE,DE,enerf,szdiag,sxdiag,nphdiag_cell,sz,sx,sy,IPR,IPR_av,eta0] = ergodic_ipr_av_new(N,cutoff,nval,OmegaR,omegaz,omegaAx,0,0);
out   = [out;[omegaz, OmegaR, IPR_av, IPR]];
disp(['wz, wra, IPR= ',num2str(omegaz), '  ',num2str(OmegaR),'  ',num2str(IPR),'  ',num2str(IPR_av)])
end
end
export=table(out);
filename=['/home/gclos/Dokumente/QSIM/Paula/SpinBoson/sbpaper/data/' num2str(N) 'ions_ipr_av_3d_n1_nc6.txt']
writetable(export,filename,'Delimiter','tab')
toc;
%took 60000s
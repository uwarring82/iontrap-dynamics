tic;
figure()
cutoff=6;
N=3;
nval = 100*[0.01,0.01,0.01,0.01,0.01];
out = [];
%for omegaAx=[0.2,0.25,0.3:0.1:1,1.5,2:2:10,100,1000]
omegaAx=0.707;
for OmegaR=-0.2:0.02:2.7;
for omegaz=-0.2:0.02:2.7;
[meanE,DE,enerf,szdiag,sxdiag,nphdiag_cell,sz,sx,sy,IPR,eta0] = ergodic_ipr_gc(N,cutoff,nval,OmegaR,omegaz,omegaAx,0,0);
out   = [out;[omegaz, OmegaR, IPR]];
disp(['wz, wra, IPR= ',num2str(eta0), '  ',num2str(OmegaR),'  ',num2str(IPR)])
end
end
export=table(out);
filename=['/home/gclos/Dokumente/QSIM/Paula/SpinBoson/' num2str(N) 'ions_ipr_3d_n1_nc6.txt']
writetable(export,filename,'Delimiter','tab')
toc;
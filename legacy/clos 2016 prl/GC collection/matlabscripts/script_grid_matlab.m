for omegaAx = 0.5:0.5:1
% N = 1    
% sval = []; 
% varval = [];
% ETHval = [];
% nval = [0.5];
% varstart = 0;
% varend = 4*omegaAx;
% varstep = varend/40;
% for omegaR = varstart:varstep:varend
% for omegaz = varstart:varstep:varend
% [P,nph,meansz,meansx,meansy,varsz,varsx,varsy] = sb_evolution_modes_Bloch(N,1,3,nval,omegaR,omegaz,omegaAx,0,0,0.0,200,1);
% [meanE,DE,enerf,szdiag,sxdiag,sz,sx,sy,ener] = ergodic(N,3,nval,omegaR,omegaz,omegaAx,0,0);
% sval   = [sval;[omegaz,omegaR,meansz]];
% varval = [varval;[omegaz,omegaR,varsz]];
% ETHval = [ETHval;[omegaz,omegaR,sz]];
% disp(['omegaz,omegaR = ',num2str(omegaz),'  ',num2str(omegaR)])
% end
% end
% filename = strcat('N',num2str(N),'Ax',num2str(omegaAx),'nval',num2str(nval(1)))
% save(filename)
N = 2    
sval = []; 
varval = [];
ETHval = [];
nval = [0.5,0.5];
varstart = 0;
varend = 4*omegaAx;
varstep = varend/40;
omegaRvec = varstart:varstep:varend;
omegazvec = varstart:varstep:varend;
sval = meshgrid(omegaRvec,omegazvec);
varval = sval;
ETHval = sval;
for i=1:length(omegaRvec)
omegaR = omegaRvec(i)
for j=1:length(omegazvec)
omegaz = omegazvec(j)
[P,nph,meansz,meansx,meansy,varsz,varsx,varsy] = sb_evolution_modes_Bloch(N,1,3,nval,omegaR,omegaz,omegaAx,0,0,0.0,200,1);
[meanE,DE,enerf,szdiag,sxdiag,sz,sx,sy,ener] = ergodic(N,3,nval,omegaR,omegaz,omegaAx,0,0);
sval(i,j)   = meansz;
varval(i,j) = varsz;
ETHval(i,j) = sz;
disp(['omegaz,omegaR = ',num2str(omegaz),'  ',num2str(omegaR)])
end
end
filename = strcat('N',num2str(N),'Ax',num2str(10*omegaAx),'nval0',num2str(10*nval(1)),'matlab.mat')
save(filename)
figure
subplot(1,3,1)
contour(omegaRvec,omegazvec,sval)
subplot(1,3,2)
contour(omegaRvec,omegazvec,sval-ETHval)
subplot(1,3,3)
contour(omegaRvec,omegazvec,varval)
end
% N = 3    
% sval = []; 
% varval = [];
% ETHval = [];
% nval = [0.5,0.5,0.5];
% varstart = 0;
% varend = 4*omegaAx;
% varstep = varend/40;
% for omegaR = varstart:varstep:varend
% for omegaz = varstart:varstep:varend
% [P,nph,meansz,meansx,meansy,varsz,varsx,varsy] = sb_evolution_modes_Bloch(N,1,3,nval,omegaR,omegaz,omegaAx,0,0,0.0,200,1);
% [meanE,DE,enerf,szdiag,sxdiag,sz,sx,sy,ener] = ergodic(N,3,nval,omegaR,omegaz,omegaAx,0,0);
% sval   = [sval;[omegaz,omegaR,meansz]];
% varval = [varval;[omegaz,omegaR,varsz]];
% ETHval = [ETHval;[omegaz,omegaR,sz]];
% disp(['omegaz,omegaR = ',num2str(omegaz),'  ',num2str(omegaR)])
% end
% end
% filename = strcat('N',num2str(N),'Ax',num2str(omegaAx),'nval',num2str(nval(1)))
% save(filename)
% end


% filename = strcat('N',num2str(N),'Ax',num2str(omegaAx),'nval',num2str(nval(1)))
% load()
% figure
% ax1 = subplot(1,2,1)
% tri1 = delaunay(sval(:,1),sval(:,2))
% h1 = trisurf(tri1, sval(:,2), sval(:,1), sval(:,3)-ETHval(:,3));
% colorbar
% hold on
% map = colormap; % current colormap
% for i=1:1:6
% map(i,:) = [1,0,0];
% end
% colormap(ax1,map)
% trisurf(tri,sval(:,1), sval(:,2), 0.1+0*sval(:,3))
% axis vis3d
% xlabel('\Omega_{Rabi} (2\pi MHz)')
% ylabel('\omega_z (2\pi MHz)')
% title('N = 3 ions, \theta = 0, \omega_{axial} = 2\pi 1.0 MHz')
% ax2 = subplot(1,2,2)
% tri2 = delaunay(varval(:,1),varval(:,2))
% h2 = trisurf(tri2, varval(:,2), varval(:,1), varval(:,3)-ETHval(:,3));
% colorbar
% hold on
% map = colormap; % current colormap
% for i=1:1:20
% map(i,:) = [1,0,0];
% end
% colormap(ax2,map)
% trisurf(tri,varval(:,1), varval(:,2), 0.25+0*varval(:,3))
% axis vis3d
% xlabel('\Omega_{Rabi} (2\pi MHz)')
% ylabel('\omega_z (2\pi MHz)')
% title('N = 3 ions, \theta = 0, \omega_{axial} = 2\pi 1.0 MHz')
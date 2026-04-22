for omegaAx = 0.5:0.5:0.5;
N = 2;
nval = [0.5];
filename = strcat('N',num2str(N),'Ax',num2str(10*omegaAx),'nval0',num2str(10*nval(1)),'matlab.mat')
load(filename)
lines=[-0.2:0.05:1];
figure
% subplot(1,3,1)
% [C,h]=contourf(omegaRvec,omegazvec,transpose(sval),lines)
% clabel(C)
% colorbar
% xlabel('\Omega_{Rabi} (2\pi MHz)')
% ylabel('\omega_z (2\pi MHz)')
% title(strcat('\langle \sigma_z \rangle, ',num2str(N),' ions, \theta = 0, \omega_{axial} = 2\pi ',num2str(omegaAx),' MHz'))
subplot(1,2,1)
[C1,h1]=contourf(omegaRvec,omegazvec,transpose(sval-ETHval),lines)
colorbar
clabel(C1)
xlabel('\Omega_{Rabi} (2\pi MHz)')
ylabel('\omega_z (2\pi MHz)')
title(strcat('\langle \sigma_z \rangle-ETH, ',num2str(N),' ions, \theta = 0, \omega_{axial} = 2\pi ',num2str(omegaAx),' MHz'))
subplot(1,2,2)
[C2,h2]=contourf(omegaRvec,omegazvec,transpose(varval),lines)
colorbar
clabel(C2)
xlabel('\Omega_{Rabi} (2\pi MHz)')
ylabel('\omega_z (2\pi MHz)')
title(strcat('var \sigma_z',num2str(N),' ions, \theta = 0, \omega_{axial} = 2\pi ',num2str(omegaAx),' MHz'))
end
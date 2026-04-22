waxvals = [];
for wax = 0.1:0.2:2.0
for N = 1:1
modes = [1.,1.73,2.48];
optmode = mean(modes(1:N)) * wax;
sval = []; 
varval = [];
ETHval = [];
nval=[];
for i = 1:N
nval = [nval;[0.5]];
end
steps = 10;
wrabivec = wax:(wax/steps):(2*wax);
wzvec = 0:(optmode*wax/steps):(optmode*wax)
sval = meshgrid(wrabivec,wzvec);
varval = sval;
ETHval = sval;
for i = 1 : length(wzvec)
wz = wzvec(i)
for j = 1 : length(wrabivec)
wrabi = wrabivec(j)
[P,nph,meansz,meansx,meansy,varsz,varsx,varsy] = sb_evolution_modes_Bloch(N,1,3,nval,wrabi,wz,wax,0,0,0.0,200,1);
[meanE,DE,enerf,szdiag,sxdiag,sz,sx,sy,ener] = ergodic(N,3,nval,wrabi,wz,wax,0,0);
sval(i,j)   = meansz;
varval(i,j) = varsz;
ETHval(i,j) = sz;
disp(['wz,wrabi = ',num2str(wz),'  ',num2str(wrabi)])
end
end
filename = strcat('N',num2str(N),'Ax',num2str(10*wax),'nval0',num2str(10*nval(1)),'integrals.mat')
save(filename)
waxvals = [waxvals;[wax,mean(abs(sval(:)-ETHval(:))),mean(varval(:))]]
figure
subplot(1,2,1)
contourf(wrabivec,wzvec,sval-ETHval)
xlabel('\Omega_{Rabi}')
ylabel('\omega_z')
title(strcat(num2str(N),' ions, wax=',num2str(wax),', \sigma_z-ETH'))
colorbar
subplot(1,2,2)
contourf(wrabivec,wzvec,varval)
xlabel('\Omega_{Rabi}')
ylabel('\omega_z')
title('var(\sigma_z)')
colorbar
end
end
figure
subplot(1,2,1)
plot(waxvals(:,1),waxvals(:,2))
xlabel('\omega_{ax}')
ylabel('\sigma_z - ETH')
title(strcat(num2str(N),' ions',', \sigma_z-ETH'))
subplot(1,2,2)
plot(waxvals(:,1),waxvals(:,3))
xlabel('\omega_{ax}')
ylabel('var(\sigma_z)')
title(strcat(num2str(N),' ions',', var(\sigma_z)'))
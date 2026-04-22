for N = 1:1:3;
eners = [1.,1.73,2.48];
optfactor = mean(eners(1:N));
sval = []; 
varval = [];
ETHval = [];
omegaAx = 0.7;
OmegaR = optfactor * omegaAx;
vsta = 0;
vend = 2.5*omegaAx;
vstep = (vend-vsta)/60;
nval=[];
for i=1:1:N
nval = [nval;[0.5]];
end
for omegaz = vsta:vstep:vend
[P,nph,meansz,meansx,meansy,varsz,varsx,varsy] = sb_evolution_modes_Bloch(N,1,3,nval,OmegaR,omegaz,omegaAx,0,0,0.0,200,1);
[meanE,DE,enerf,szdiag,sxdiag,sz,sx,sy,ener] = ergodic(N,3,nval,OmegaR,omegaz,omegaAx,0,0);
sval   = [sval;[omegaz,meansz,meansx,meansy]];
varval = [varval;[omegaz,varsz,varsx,varsy]];
ETHval = [ETHval;[omegaz,sz,sx,sy]];
disp(['omegaz = ',num2str(omegaz)])
end
figure
subplot(1,2,1)
hold on
plot(sval(:,1),sval(:,2),'k','Linewidth',2)
plot(sval(:,1),ETHval(:,2),'k','Linewidth',3)
plot(sval(:,1),sval(:,3)-ETHval(:,3),'r','Linewidth',2)
plot(sval(:,1),sval(:,4)-ETHval(:,4),'g','Linewidth',2)
legend('\langle \sigma_z \rangle - \langle \sigma_z \rangle_{ETH}',...
       '\langle \sigma_x \rangle - \langle \sigma_x \rangle_{ETH}',...
       '\langle \sigma_y \rangle - \langle \sigma_y \rangle_{ETH}')
xlabel('\omega_z (MHz)')
title(strcat(num2str(N),'ions, \omega_{axial} = 2\pi ',num2str(omegaAx),' MHz,',' \Omega_R=',num2str(OmegaR),'MHz'))
grid on
subplot(1,2,2)
hold on
plot(sval(:,1),varval(:,2),'k','Linewidth',2)
plot(sval(:,1),varval(:,3),'r','Linewidth',2)
plot(sval(:,1),varval(:,4),'g','Linewidth',2)
legend('Var \langle \sigma_z \rangle',...
       'Var \langle \sigma_x \rangle',...
       'Var \langle \sigma_y \rangle')
xlabel('\omega_z (MHz)')
grid on
end
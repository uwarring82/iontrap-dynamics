figure
hold on
for tsta = 1:2:10
N = 2 ;
sval = []; 
varval = [];
ETHval = [];
omegaAx = 0.708;
vsta = 0;
vend = 2.0;
vstep = (vend-vsta)/30;
nval=[0.5,0.5];
OmegaR = 0.966;
for omegaz = vsta:vstep:vend
[P,nph,meansz,meansx,meansy,varsz,varsx,varsy] = sb_evolution_modes_Bloch(N,1,ncut,nval,OmegaR,omegaz,omegaAx,0,0,tsta,60,0.5);
[meanE,DE,enerf,szdiag,sxdiag,sz,sx,sy,ener] = ergodic(N,3,nval,OmegaR,omegaz,omegaAx,0,0);
sval   = [sval;[omegaz,meansz,meansx,meansy]];
varval = [varval;[omegaz,varsz,varsx,varsy]];
ETHval = [ETHval;[omegaz,sz,sx,sy]];
disp(['omegaz = ',num2str(omegaz)])
end
subplot(1,2,1)
hold on
plot(sval(:,1),sval(:,2),'k','Linewidth',2)
plot(sval(:,1),ETHval(:,2),'k','Linewidth',3)
%plot(sval(:,1),sval(:,3)-ETHval(:,3),'r','Linewidth',2)
%plot(sval(:,1),sval(:,4)-ETHval(:,4),'g','Linewidth',2)
%legend('\langle \sigma_z \rangle - \langle \sigma_z \rangle_{ETH}',...
%       '\langle \sigma_x \rangle - \langle \sigma_x \rangle_{ETH}',...
%       '\langle \sigma_y \rangle - \langle \sigma_y \rangle_{ETH}')
xlabel('\omega_z (MHz)')
title(strcat('\langle \sigma_z \rangle, ',num2str(N),'ions, \theta = 0, \omega_{axial} = 2\pi ',num2str(omegaAx),' MHz'))
grid on
subplot(1,2,2)
hold on
plot(sval(:,1),varval(:,2),'k','Linewidth',2)
%plot(sval(:,1),varval(:,3),'r','Linewidth',2)
%plot(sval(:,1),varval(:,4),'g','Linewidth',2)
%legend('Var \langle \sigma_z \rangle',...
%       'Var \langle \sigma_x \rangle',...
%       'Var \langle \sigma_y \rangle')
xlabel('\omega_z (MHz)')
grid on
end
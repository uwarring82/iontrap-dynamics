N = 2;
sval = []; 
varval = [];
ETHval = [];
nval = [0.5,0.5];
omegaAx = 0.7;
omegaR = 1.35 * omegaAx;
for omegaz = 0.02:0.03:2
[P,nph,meansz,meansx,meansy,varsz,varsx,varsy] = sb_evolution_modes_Bloch(N,1,3,nval,omegaR,omegaz,omegaAx,0,0,0.0,200,1);
[meanE,DE,enerf,szdiag,sxdiag,sz,sx,sy,ener] = ergodic(N,3,nval,omegaR,omegaz,omegaAx,0,0);
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
legend('\langle \sigma_z \rangle',...
       '\langle \sigma_z \rangle_{ETH}',...
       '\langle \sigma_x \rangle - \langle \sigma_x \rangle_{ETH}',...
       '\langle \sigma_y \rangle - \langle \sigma_y \rangle_{ETH}')
xlabel('\omega_z (MHz)')
nstr = num2str(N);
orstr = num2str(omegaR)
oaxstr = num2str(omegaAx)
title(strcat('N =',nstr,'ions, \theta = 0, \Omega_R =',orstr,' MHz, \omega_{axial} = ',oaxstr,'MHz'))
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
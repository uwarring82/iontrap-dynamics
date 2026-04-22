% svals=[];
% varvals=[];
% ETHvals=[];
% for N = 1:1:3;    
% eners = [1.,1.73,2.41];
% optfactor = mean(eners(1:N));
% sval = [];
% varval = [];
% ETHval = [];
% nval=[];
% for i=1:1:N
% nval = [nval;[0.5]];
% end
% for omegaAx = [0.01:0.01:0.1,0.12:0.02:0.3,0.35:0.05:2]
% OmegaR = optfactor * omegaAx;
% omegaz = optfactor * omegaAx;
% [P,nph,meansz,meansx,meansy,varsz,varsx,varsy] = sb_evolution_modes_Bloch(N,1,3,nval,OmegaR,omegaz,omegaAx,0,0,0.0,200,1);
% [meanE,DE,enerf,szdiag,sxdiag,sz,sx,sy,ener] = ergodic(N,3,nval,OmegaR,omegaz,omegaAx,0,0);
% sval   = [sval;[omegaAx,meansz,meansx,meansy]];
% varval = [varval;[omegaAx,varsz,varsx,varsy]];
% ETHval = [ETHval;[omegaAx,sz,sx,sy]];
% disp(['omegaz = ',num2str(omegaAx)])
% end
% svals = [svals,[sval]];
% ETHvals = [ETHvals,[ETHval]];
% varvals = [varvals,[varval]];
% save('axials_n123_allinone.mat')
% end

load('axials_n123_allinone.mat')
t=figure('units','normalized','outerposition',[0 0 1 1])
ha = tight_subplot(1,3,[.06 .06],[.1 .1],[.06 .06])
axes(ha(1))
plot(svals(:,1),svals(:,2))
hold on
plot(svals(:,5),svals(:,6))
plot(svals(:,9),svals(:,10))
line([0,2],[0,0],'Color','k')
xlabel('\omega_{axial} (2\pi MHz)')
title('mean(\sigma_z)')
legend('N=1','N=2','N=3')
axis([0,2,-0.3,0.5])
axis square
axes(ha(2))
plot(svals(:,1),svals(:,2)-ETHvals(:,2))
hold on
plot(svals(:,5),svals(:,6)-ETHvals(:,6))
plot(svals(:,9),svals(:,10)-ETHvals(:,10))
line([0,2],[0,0],'Color','k')
xlabel('\omega_{axial} (2\pi MHz)')
title('mean(\sigma_z) - ETH')
legend('N=1','N=2','N=3')
axis([0,2,-0.1,0.3])
axis square
axes(ha(3))
plot(varvals(:,1),varvals(:,2))
hold on
plot(varvals(:,5),varvals(:,6))
plot(varvals(:,9),varvals(:,10))
xlabel('\omega_{axial} (2\pi MHz)')
title('var(\sigma_z)')
legend('N=1','N=2','N=3')
axis([0,2,0,0.5])
axis square
subtitle(strcat('\omega_{z} = \omega_{Rabi} = optimal'))
set(findall(gcf,'type','axes'),'fontsize',18)
set(findall(gcf,'type','text'),'fontSize',18)
set(0,'defaultlinelinewidth',4)
export_fig -painters -r600 -q101 /home/gclos/Schreibtisch/Diego_results/axials_n123.pdf

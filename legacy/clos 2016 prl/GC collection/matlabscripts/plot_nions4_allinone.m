load('nions4_allinone_nbar05_t10omega_opt.mat')
t=figure('units','normalized','outerposition',[0 0 1 1])
ha = tight_subplot(1,3,[.06 .06],[.1 .1],[.06 .06])
axes(ha(1))
plot(svals(:,1),svals(:,2))
hold on
plot(svals(:,5),svals(:,6))
plot(svals(:,9),svals(:,10))
plot(svals(:,13),svals(:,14))
xlabel('\omega_z (2\pi MHz)')
title('mean(\sigma_z)')
legend('N=1','N=2','N=3')
axis([0,2.5,-0.3,0.7])
axis square
line([0,2],[0,0],'Color','k')
axes(ha(2))
plot(svals(:,1),svals(:,2)-ETHvals(:,2))
hold on
plot(svals(:,5),svals(:,6)-ETHvals(:,6))
plot(svals(:,9),svals(:,10)-ETHvals(:,10))
plot(svals(:,13),svals(:,14)-ETHvals(:,14))
xlabel('\omega_z (2\pi MHz)')
title('mean(\sigma_z) - ETH')
legend('N=1','N=2','N=3')
axis([0,2.5,-0.5,0.7])
axis square
line([0,2],[0,0],'Color','k')
axes(ha(3))
plot(varvals(:,1),varvals(:,2))
hold on
plot(varvals(:,5),varvals(:,6))
plot(varvals(:,9),varvals(:,10))
plot(varvals(:,13),varvals(:,14))
xlabel('\omega_z (2\pi MHz)')
title('var(\sigma_z)')
legend('N=1','N=2','N=3','N=4')
axis([0,2.5,-0,0.6])
axis square
subtitle(strcat('\omega_{axial} = 2\pi ',num2str(omegaAx),' MHz,  ',' \omega_R',' optimal'))
set(findall(gcf,'type','axes'),'fontsize',18)
set(findall(gcf,'type','text'),'fontSize',18)
set(0,'defaultlinelinewidth',4)
save '/home/gclos/Dokumente/QSIM/Paula/SpinBoson/thesis_gclos/matlabexport.txt' svals -ASCII
export_fig -painters -r600 -q101 /home/gclos/Schreibtisch/Diego_results/nions_nbar05.pdf


tic;
figure
hold all
for tmax=[15,30];
store = [];
ncut=5;
omegaAx = 0.708;
nval = [0.5,0.5,0.5,0.5,0.5];
wrabis = [0.73,0.95,1.28,1.37,1.58];
tsta = 0;
for N = 1:1:4;
OmegaR = wrabis(N);
omegaz = wrabis(N);
%tend = tmax * 2 * pi;
tend = tmax * (1/OmegaR) * 2 * pi;
tstep = (tend - tsta) / (tmax*10);
[P,nph,meansz,meansx,meansy,varsz,varsx,varsy] = sb_evolution_modes_Bloch(N,1,ncut,nval,OmegaR,omegaz,omegaAx,0,0,tsta,tend,tstep);
disp(['omegaz = ',num2str(omegaz)])
store = [store;[N,meansz,varsz,tsta,tend,tstep,ncut,omegaAx,OmegaR,omegaz,nval]];
end
plot(store(:,1),store(:,3),'Linewidth',2)
xlabel('number of ions')
ylabel('stddev')
title('natural integration time 15,30 periods')
end
export = table(store);
filename = ['/home/gclos/Dokumente/QSIM/Paula/SpinBoson/int5_natural_' num2str(tmax) '.txt']
writetable(export, filename, 'Delimiter', 'tab')
toc;
hold off

tic;
figure
hold all
for tmax=[15,30];
store = [];
ncut=5;
omegaAx = 0.708;
nval = [0.5,0.5,0.5,0.5,0.5];
wrabis = [0.73,0.95,1.28,1.37,1.58];
tsta = 0;
for N = 1:1:4;
OmegaR = wrabis(N);
omegaz = wrabis(N);
tend = tmax * 2 * pi;
%tend = tmax * (1/OmegaR) * 2 * pi;
tstep = (tend - tsta) / (tmax*10);
[P,nph,meansz,meansx,meansy,varsz,varsx,varsy] = sb_evolution_modes_Bloch(N,1,ncut,nval,OmegaR,omegaz,omegaAx,0,0,tsta,tend,tstep);
disp(['omegaz = ',num2str(omegaz)])
store = [store;[N,meansz,varsz]];
end
plot(store(:,1),store(:,3),'Linewidth',2)
xlabel('number of ions')
ylabel('stddev')
title('fixed integration time 15,30us')
end
export = table(store);
filename = ['/home/gclos/Dokumente/QSIM/Paula/SpinBoson/int5_fixed_' num2str(tmax) '.txt']
writetable(export, filename, 'Delimiter', 'tab')
toc;
hold off
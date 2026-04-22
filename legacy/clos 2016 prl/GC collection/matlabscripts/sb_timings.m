nval = [0.5 0.5 0.5 0.5 0.5];
OmegaR = 1.;
omegaz = 0.;
omegaAx = 1.;
tsta = 10;
tend = 50;
tstep = (tend-tsta) / 1;
alltimings = [];
for N = 1:5
    for cutoff=1:6
        tic;
        [P,nph,meansz,meansx,meansy,varsz,varsx,varsy] = sb_evolution_modes_Bloch(N,1,cutoff,nval,OmegaR,omegaz,omegaAx,0,0,tsta,tend,tstep);
        eltime = toc;
        alltimings = [alltimings ; [N, cutoff, eltime]]
        export = table(alltimings);
        filename = ['/home/gclos/Dokumente/QSIM/Paula/SpinBoson/timings.txt']
        writetable(export, filename, 'Delimiter', 'tab')
    end
end
disp(num2str(alltimings))
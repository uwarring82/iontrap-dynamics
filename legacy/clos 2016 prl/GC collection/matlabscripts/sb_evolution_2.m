function [P,nph,meansz,varsz,Vs,On_cell,H] = sb_evolution(eta0,N,nc,T,Omega,omegaz,axial,tmin,tmax,tstep)
%
center = 1;
[Xmin,ener,center_wf,V] = cchain(N,center);
%
ener = ener*axial;
%
[Oa_cell,On_cell,Oid_cell,Osz,Osm,Osp] = Operators(nc,N);
%
Id = speye(length(On_cell{1}));
%center = round(N/2);
%
Hsph = sparse(0);
pol = sparse(0);
Hn = sparse(0);
for n=1:N
    pol = pol + i*eta0*center_wf(n)*(1/sqrt(ener(n)/ener(1)))*(Oa_cell{n} + Oa_cell{n}');
%    pol = pol + i*eta0*center_wf(n)*(1/sqrt(ener(n)))*(Oa_cell{n} + Oa_cell{n}');
    Hn = Hn + ener(n)*On_cell{n};
end
%
Hsph = (Omega/2)*Osp*expm(pol);
Hsph = Hsph + Hsph';
%
for n = 1:N
Z(n) = sum(exp(-ener(n)*[0:nc]/T));
end
%
H = Hsph + Hn  + (omegaz/2)*Osz;
%
% build the initial thermal denisty matrix
%
rho0 = 1;
for n = 1:N
   rho0 = (1/Z(n))*expm(-ener(n)*On_cell{n}/T)*rho0;
end
%
%%%% do if you want the system to be initialized in the HS eigenbasis CHECK
%%%% IT ITS WRONG!!
HS0 = (Omega/2)*(Osm+Osp) + (omegaz/2)*Osz + 0*Hn;
[Vs,Ds] = eig(full(HS0));
%rho0 = Vs*rho0*(Osz-Id)*Vs'/2;
rho0 = rho0*(Osz+Id)/2;
%
rho0 = rho0/trace(rho0);
trace(rho0)
count = 1;
%
[VV,DD] = eig(full(H));
DD = diag(DD);
for t = tmin:tstep:tmax
    %%Psip = sparse((Id - 0.5*i*tstep*H)*Psi);
    %%Psi = Psip\(Id + 0.5*i*H*tstep);
    %%Psi = Psi';
    %U = expm(-i*H*t);
    U = VV*diag(exp(-i*DD*t))*VV';
    rho = U*rho0*U';
    % conver time to microseconds
    P(1,count) = t/(2*pi);
    P(2,count) = real(trace(rho*Osz));
    P(3,count) = real(trace(rho*Oa_cell{center}*Osp)-trace(rho*Oa_cell{center})*trace(rho*Osp));
    rhoS(1,count) = real(trace(rho*(Id + Osz)))/2;
    rhoS(4,count)= real(trace(rho*(Id - Osz)))/2;
    rhoS(2,count) = trace(rho*Osm);
    rhoS(3,count) = trace(rho*Osp);
    for n = 1:N
        nph(count,n) = real(trace(rho*On_cell{n}));
    end
        count = count + 1;
disp(t)
end
%
%
rhoAV1 = sum(rhoS(1,:))/length(rhoS(1,:));
rhoAV2 = sum(rhoS(2,:))/length(rhoS(2,:));
rhoAV3 = sum(rhoS(3,:))/length(rhoS(3,:));
rhoAV4 = sum(rhoS(4,:))/length(rhoS(4,:));
%
rhoAV = [[rhoAV1,rhoAV2];[rhoAV3,rhoAV4]];
%
%
count = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for t = tmin:tstep:tmax
% %
%     rhoD1 = rhoS(1,count) - rhoAV1;
%     rhoD2 = rhoS(2,count) - rhoAV2;
%     rhoD3 = rhoS(3,count) - rhoAV3;
%     rhoD4 = rhoS(4,count) - rhoAV4;
%       rhoDmat = [[rhoD1,rhoD2];[rhoD3,rhoD4]];
%       rhoDmat_diag = [[rhoD1,0];[0,rhoD4]];
%       Drho(1,count) = t/(2*pi);
%       %Drho(2,count) = real(0.5*trace(sqrtm(rhoDmat^2)));
%       %Drho(3,count) = 0.5*trace(sqrtm(rhoDmat_diag^2));
%       count = count + 1;
% end
% %DrhoAV = real(sum(Drho(2,:))/length(Drho(2,:)));
% %DrhoAV_diag =sum(Drho(3,:))/length(Drho(2,:));
% %
% DrhoAV_bound = 1;
% for n = 1:N
%     x = exp(-ener(n)/T);
%     f = (1-x)/(1+x);
%     DrhoAV_bound = DrhoAV_bound*sqrt(f);
% end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
meansz = sum(P(2,:))/length(P(2,:));
varsz = sqrt(sum((P(2,:)-meansz).^2)/length(P(2,:)));
end
%
function [Oa_cell,On_cell,Oid_cell,Osz,Osm,Osp] = Operators(nc,Nsites)
%
n = 1;
j = n/2;
m_vec = -n/2+1:n/2;
sm = sparse(diag(sqrt(j*(j+1)-m_vec.*(m_vec-1)),-1)); 
sp = sm';
sz = 2*sparse(diag(n/2:-1:-n/2));
%sx = (sm + sp);
%
%sy = i*(sm - sp);
a = sparse(diag(sqrt([1:nc]),1)); 
%nn =  sparse(diag([0:nc]));
nn = a'*a;
is = speye(n+1);
ip = speye(nc+1);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
Oa_cell = cell(Nsites,1);
On_cell = cell(Nsites,1);
Oid_cell = cell(Nsites,1);
%Osz_cell = cell(Nsites,1);
%Osm_cell = cell(Nsites,1);
%Osp_cell = cell(Nsites,1);
%
for j=1:Nsites
    Oa_cell{j} = 1;
    On_cell{j} = 1;
    Oid_cell{j} = 1;
        for jj = 1:j-1
            Oa_cell{j} = kron(ip,Oa_cell{j});
            On_cell{j} = kron(ip,On_cell{j});
            Oid_cell{j} = kron(ip,Oid_cell{j});
        end
            Oa_cell{j} = kron(a,Oa_cell{j});
            On_cell{j} = kron(nn,On_cell{j});
            Oid_cell{j} = kron(ip,Oid_cell{j});
        for jj = j+1:Nsites
            Oa_cell{j} = kron(ip,Oa_cell{j});
            On_cell{j} = kron(ip,On_cell{j});
            Oid_cell{j} = kron(ip,Oid_cell{j});
        end
end
%
Iph = speye(length(Oa_cell{1}));
%
for j=1:Nsites
    Oa_cell{j} = kron(is,Oa_cell{j});
    On_cell{j} = kron(is,On_cell{j});
    Oid_cell{j} = kron(is,Oid_cell{j});
end
%
Osz = kron(sz,Iph);
Osp = kron(sp,Iph);
Osm = kron(sm,Iph);
end
%
function eta = eta_calculator(m,lam,wx)
% m atomic units
% lam nanometers
% wx MHz
hbar = 1.05457173*10^(-34);
mm = m*1.66*10^(-27);
wwx = 2*pi*wx*10^6;
k = 2*pi/(lam*10^(-9));
%
eta = sqrt(hbar/(2*mm*wwx))*k;
end
%

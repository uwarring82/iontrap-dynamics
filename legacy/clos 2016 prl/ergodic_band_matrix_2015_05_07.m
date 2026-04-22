function [IPR] = ergodic_band_matrix(N,nc,n_val,Omega,omegaz,axial)
%
if N < 4
    Nstatesmin = 5000;
else
    if N < 5
    Nstatesmin = 8000;
    else
    Nstatesmin = 20000; end
end
%
% Nstatesmin is the maximum size of the blocks in which the Hamiltonian is
% divided - one can play with it to assess the complexity of the
% Hamiltonian!!
%
disp(['Maximum size of Hamiltonian Blocks = ' num2str(Nstatesmin)])
warning('OFF'); 
nphdiag_cell = cell(1,N);
%
% calculate vibrational eigenmodes / COM Lamb-Dicke parameter - assumes
% spin in position 1 in the chain
center = 1;
[Xmin,ener,center_wf,V] = cchain(N,center);
%
ener = ener*axial;
% CALCULATION WITH SPIN-BOSON COUPLING
eta0 = eta_calculator(25,279.63/sqrt(2.),axial);
%
%
% define operators
[Oa_cell,On_cell,Oid_cell,Osz,Osm,Osp] = Operators(nc,N);
Osx = Osp + Osm; Osy = -i*(Osp - Osm);
%
Id = speye(length(On_cell{1}));
%;
% build Hamiltonian
Hsph = sparse(0);
pol = sparse(1);
Hn = sparse(0);
%
for n=1:N
    Hn = Hn + ener(n)*On_cell{n};
end
%
th = atan(Omega/omegaz);
H0 = Hn + (Omega/2)*Osx + (omegaz/2)*Osz;
clear('Hn');
%
[dH0,ind] = sort(diag(H0));
clear('dH0');
maxmat = min(120000,length(H0)); 
H0 = H0(ind(1:maxmat),ind(1:maxmat));
%
for n =1:N
    pol = pol*(Id + eta0*center_wf(n)*(1/sqrt(ener(n)/ener(1)))*(Oa_cell{n} - Oa_cell{n}') + 0.5*(eta0*center_wf(n)*(1/sqrt(ener(n)/ener(1)))*(Oa_cell{n} - Oa_cell{n}'))^2 + (1/6)*(eta0*center_wf(n)*(1/sqrt(ener(n)/ener(1)))*(Oa_cell{n} - Oa_cell{n}'))^3 + (1/(6*4))*(eta0*center_wf(n)*(1/sqrt(ener(n)/ener(1)))*(Oa_cell{n} - Oa_cell{n}'))^4);
    %whos('pol')
    %pol = pol*(Id + eta0*center_wf(n)*(1/sqrt(ener(n)/ener(1)))*(Oa_cell{n} - Oa_cell{n}') + 0.5*(eta0*center_wf(n)*(1/sqrt(ener(n)/ener(1)))*(Oa_cell{n} - Oa_cell{n}'))^2);
end
%
Hsph = (Omega/2)*(Osp)*(pol-speye(length(Id)));
clear('pol');
Hsph = Hsph + Hsph';
Hsph = Hsph(ind(1:maxmat),ind(1:maxmat));
% calculate the intitial density matrix
%
for n = 1:N
    T(n) = ener(n)/log(1+1/n_val(n));
    %Z(n) = sum(exp(-ener(n)*[0:nc]/T(n)));    
    Z(n) = sum(exp(-ener(n)*[0:100]/T(n)));
end
%
rho0 = 1;
%
for n = 1:N
   rho0 = (1/Z(n))*diag(exp(-ener(n)*diag(On_cell{n})/T(n)))*rho0;
end
%
rho0 = rho0*(Id + Osz)/2;
%
rho0 = diag(rho0(ind(1:maxmat),ind(1:maxmat)));
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CALCULATION OF THE AVERAGE IPR %%
%
nstep = Nstatesmin/4;
%nstep = 1;
%
if Nstatesmin + nstep + 1 < length(H0)
%%
Nstates = min(Nstatesmin,length(H0));
%
disp(['from j = ' num2str(Nstates/2) ' to '  num2str(length(H0) - (Nstates/2))])
deffav = [];
enerav = [];
%
    HProy = H0(1:Nstates,1:Nstates) + Hsph(1:Nstates,1:Nstates);
    HProy = (HProy + HProy')/2;
%
    [VProy,DProy] = eig(full(HProy));
    deffvec = 1./sum(VProy(:,:)'.^4);
    deffav = deffvec(1:Nstates/2+nstep/2);
    enerav = diag(full(H0(1:Nstates/2+nstep/2,1:Nstates/2+nstep/2)));
%
%deff0 = full(sum(deffav'.*rho0(1:length(enerav))));
Nsteps = round((length(H0) - (Nstates/2 + nstep) - Nstates/2)/nstep);
count = 1;
defflist = [];
%
deff0 = 0;
for j = Nstates/2+nstep+1:nstep:length(H0)-(Nstates/2)
    tic
    nmin = j - Nstates/2;
    nmax = j + Nstates/2+1;
    HProy = H0(nmin:nmax,nmin:nmax) + Hsph(nmin:nmax,nmin:nmax);
    HProy = (HProy + HProy')/2;
%
    nj = Nstates/2;
    disp(['j = ' num2str(j)])
    [VProy,DProy] = eig(full(HProy));
    enervec = diag(full(H0(nmin:nmax,nmin:nmax)));
    deffvec = 1./sum(VProy(:,:)'.^4);
    deffav = [deffav , 1./sum(VProy(nj-(nstep/2-1):nj+nstep/2,:)'.^4)];
    enerav = [enerav ; enervec(nj-(nstep/2-1):nj+nstep/2)];
    timestep = toc/60;
    timeleft = timestep*(Nsteps - count + 1);
    disp(['time left = ' num2str(timeleft) ' minutes'])
    count = count + 1;
    deff = full(sum(deffav'.*rho0(1:length(enerav))));
    defflist = [defflist;deff];
    disp(['deff = ' num2str(deff)]);
    if abs(deff-deff0)/deff < 0.0005 
        break
    end
    deff0 = deff;
    %
end
%
% j = length(H0) - (Nstates/2)+nstep;
%     nmin = j - Nstates/2;
%     nmax = length(H0);
%     HProy = H0(nmin:nmax,nmin:nmax) + Hsph(nmin:nmax,nmin:nmax);
%     HProy = (HProy + HProy')/2;
% 
%     nj = Nstates/2 + 1;
%     disp(['j = ' num2str(j)])
%     [VProy,DProy] = eig(full(HProy));
%     enervec = diag(H0(nmin:nmax,nmin:nmax));
%     deffvec = 1./sum(VProy(:,:)'.^4);
%     deffav = [deffav , 1./sum(VProy(nj-(nstep/2-1):length(VProy),:)'.^4)];
%     enerav = [enerav ; enervec(nj-(nstep/2-1):length(VProy))];
%     timestep = toc/60;
%     timeleft = timestep*(Nsteps - count + 1);
%     disp(['time left = ' num2str(timeleft) ' minutes'])
%     count = count + 1;
%     
% % %%%%%%
%deffave_rho = full(deffav'.*rho0(1:length(enerav)));
deff = full(sum(deffav'.*rho0(1:length(enerav))));
%deff1 = deff - deff0;
else
Nstates = min(Nstatesmin,length(H0));
nstep = Nstates;
disp(['from j = ' num2str(Nstates/2) ' to '  num2str(length(H0) - (Nstates/2))])
deffav = [];
enerav = [];
%
    HProy = H0(1:Nstates,1:Nstates) + Hsph(1:Nstates,1:Nstates);
    HProy = (HProy + HProy')/2;
%
    [VProy,DProy] = eig(full(HProy));
    deffvec = 1./sum(VProy(:,:)'.^4);
    deffav = deffvec(1:Nstates/2+nstep/2);
    enerav = diag(H0(1:Nstates/2+nstep/2,1:Nstates/2+nstep/2));
%
end
deff = full(sum(deffav'.*rho0(1:length(enerav))));
%
IPR = deff;
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
%  FROM HERE ON: CODE TO CALCULATE PROPERTIES OF THE ION CHAIN: EQ.
%  POSITIONS, VIBRATIONAL ENERGIES AND PHONON WAVEFUNCTIONS
%
function eta = eta_calculator(m,lam,wx)
% CALCULATES THE LAMB-DICKE PARAMETER 
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
function [Xmin,ener,center_wf,V] = cchain(n,center)
% n = number of ions
% center = position of the spin-ion
X0 = initial(n);
options = optimset('GradObj','on','Hessian','on','TolX',10^-16,'TolFun',10^-16,'Display','off');
[Xmin fval exitflag output grad Hessian] = fminunc(@mypotential,X0,options,4,n);
[V, ener] = eig(0.5*Hessian);
ener = sqrt(diag(ener));
center_wf = V(center,:);
end
% ==========
function initial = initial(n)
%we give an estimation of the equilibrium position of the ions as initial
%condition for the numerical calculation
b = beta(n);
%d0 is the distance between ions in a0^3 = e^2/(m omega_z^2) units, as
%estimated by the variational ansatz
d0 = (2/b)^(1/3);
for i=1:n
    X(i) = (-n/2 -1/2 + i)*d0;
end
initial = X;
end
% ==============
function b=beta(n)
b = (1/12)*n^2/(0.577 + log(6*n)-13/5) ;
end
% ==============
% DEFINE FUNCTION MYPOTENTIAL
% ==============
function [f,g,H] = mypotential(X,beta,n)
f = potential(X,beta,n);
if nargout > 1
g = firstderivative(X,beta,n);
if nargout > 2 
H = hessian(X,beta,n);
end
end
end
% =================
% COULOMB POTENTIAL
% =================
function potential = potential(x,beta,n)
poten = 0 ;
coul = 0 ;
for i=1:n
    poten = poten + (x(i))^2 ;
end
for i=1:n
    for j=1:i-1
        coul = coul + 1/abs(x(i)-x(j)) ;
    end
end
coulomb = coul*(beta/2) ;
potential = poten + coulomb ;
end
% FIRST DERIVATIVES OF THE COULOMB POTENTIAL
function firstderivative = firstderivative(X,beta,n)
for i=1:n
    sum = 0;
    for j=1:i-1
        sum = sum + sign(X(i)-X(j))/(X(i)-X(j))^2;
    end
    for j=i+1:n
        sum = sum + sign(X(i)-X(j))/(X(i)-X(j))^2;
    end
    Y(i) = - (beta/2)*sum + 2*X(i);
end
firstderivative = Y;
end
% HESSIAN OF THE COULOMB POTENTIAL
function hessian = hessian(X,beta,n)
for i=1:n
    for j= 1:i-1
        Y(i,j)=-beta/abs((X(i)-X(j))^3) ;
    end
    for j= i+1:n
        Y(i,j)=-beta/abs((X(i)-X(j))^3) ;
    end
end
for i=1:n
    sum = 0;
    for j=1:i-1
        sum = sum + 1/abs((X(i)-X(j))^3) ;
    end
    for j=i+1:n
        sum = sum + 1/abs((X(i)-X(j))^3) ;
    end
    Y(i,i) = beta*sum + 2;
end
hessian = Y;
%
end
%
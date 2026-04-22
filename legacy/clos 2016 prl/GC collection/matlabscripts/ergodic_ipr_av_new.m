function [meanE,DE,enerf,szdiag,sxdiag,nphdiag_cell,sz,sx,sy,IPR,IPR_av,eta0] = ergodic(N,nc,n_val,Omega,omegaz,axial,theta,phi)
%
% 'ergodic.m' calculates mean values under the assumption that the Eigenstate Thermalization Hypothesis holds
% %%%%
% INPUT
% N = number of ions
% nc = maximum number of phonons (3-5 usually works up to Lamb-Dicke parameters \approx 1)
% n_val = array containing the mean vibrational mode occupaton numbers
%      n_val = [n1,n2,n3,..], mean phonon number in mode 1 = n1, etc...
%      (Assumes thermal phonon distribution within each mode!!)
% Omega = Two-photon Rabi Frequency (in MHz: for 'Omega' = 1 (2 pi) MHz, we enter value Omega = 1, etc.)
% omegaz = detuning with respect to the carrier (in MHz)
% axial = axial trapping frequency (in MHz)
% theta, phi = angles in the Bloch sphere of the intial state:
%    \psi = cos(theta/2) |up> + sin(theta/2) e^(-i \phi) |down>
%    e.g.: phi = 0, theta = 0: initial state pointing 'up'
%%%%
% OUTPUT
%
% meanE = mean enegy of the initial state
% DE = energy variance of the intial state
% enerf = set of energy levels of the whole Hamiltonian (spin + bath + spin-bat coupling) used in the calculation
% szdiag/sxdiag = meanvalues of \sigma^z/\sigma^x for each level in 'enerf'
% sz, sx, sy = mean values of sigma^x,y,z (sy is redundant -> always zero, since H is Real)
% IPR = Inverse Participation Ratio (effective number of states taking part in the quantum evolution)
% IPR_av = averaged participation ratio over the mixed state
%%%% the thermal initial state
%
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
% define operators
[Oa_cell,On_cell,Oid_cell,Osz,Osm,Osp] = Operators(nc,N);
Osx = Osp + Osm; Osy = -i*(Osp - Osm);
%
Id = speye(length(On_cell{1}));
%;
% build Hamiltonian
Hsph = sparse(0);
pol = sparse(0);
Hn = sparse(0);
for n=1:N
    pol = pol + eta0*center_wf(n)*(1/sqrt(ener(n)/ener(1)))*(Oa_cell{n} - Oa_cell{n}');
    Hn = Hn + ener(n)*On_cell{n};
end
%
Hsph = (Omega/2)*Osp*expm(pol);
Hsph = Hsph + Hsph';
%
H = Hsph + Hn  + (omegaz/2)*Osz;
H = (H + H')/2;
%
% calculate the intitial density matrix
%
for n = 1:N
    T(n) = ener(n)/log(1+1/n_val(n));
    Z(n) = sum(exp(-ener(n)*[0:nc]/T(n)));
end
%
rho0 = 1;
%
for n = 1:N
   rho0 = (1/Z(n))*expm(-ener(n)*On_cell{n}/T(n))*rho0;
end
%
rho0 = rho0*(Id + cos(theta)*Osz + sin(theta)*cos(phi)*Osx + sin(theta)*sin(phi)*Osy)/2;
%
% calculate <E>, var(E)
meanE = real(trace(rho0*H));
DE = sqrt(abs(trace(rho0*H^2) - meanE^2));
% calculate 300 first eigenstates
%Neigens = min(300,length(H))
%
%[VVf,enerf] = eigs(H,Neigens,meanE);
[VVf,enerf] = eig(H);
%
%
enerf = diag(enerf);
szdiag = diag(VVf(:,:)'*Osz*VVf(:,:));
sxdiag = diag(VVf(:,:)'*Osx*VVf(:,:));
%
rho_d = diag(VVf(:,:)'*rho0*VVf(:,:));
IPR = 1/sum(rho_d.^2);
%
for nn = 1:N
nphdiag_cell{nn} = diag(VVf(:,:)'*On_cell{nn}*VVf(:,:));
end
%
% CALCULATE MEAN VALUES - WEIGHTS CONTRIBUTION OF EACH EIGENSTATE BY A
% GAUSSIAN WEIGHT AROUND 'meanE' WITH WIDTH 'DE'.
%
sz = sum(diag(VVf'*Osz*VVf).*exp(-(enerf - meanE).^2/(DE/2)^2))/sum(exp(-(enerf - meanE).^2/(DE/2)^2));
sz = real(sz);
%
sx = sum(diag(VVf'*Osx*VVf).*exp(-(enerf - meanE).^2/(DE/2)^2))/sum(exp(-(enerf - meanE).^2/(DE/2)^2));
sx = real(sx);
%
sy = sum(diag(VVf'*Osy*VVf).*exp(-(enerf - meanE).^2/(DE/2)^2))/sum(exp(-(enerf - meanE).^2/(DE/2)^2));
sy = real(sy);
%
for nn=1:N
nphETH(nn) = sum(diag(VVf'*On_cell{nn}*VVf).*exp(-(enerf - meanE).^2/(DE/2)^2))/sum(exp(-(enerf - meanE).^2/(DE/2)^2));
nphETH(nn) = real(nphETH(nn));
end
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CALCULATION OF THE AVERAGE IPR %%
%
[V0,D0] = eig(rho0);

%IPR_av = 0;
% %
% for j=1:length(D0)
% rho0j = V0(:,j)*V0(:,j)';
% rho_dj = diag(VVf(:,:)'*rho0j*VVf(:,:));
% r2 = sum(rho_dj.^2);
% IPR_av = IPR_av + D0(j,j)/r2;
% end
% 
IPR_av = sum((1./sum(abs(VVf(:,:)'*V0(:,:)).^4))*D0);
%IPR_av = 1/sum((sum(abs(VVf(:,:)'*V0(:,:)).^4))*D0);
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CALCULATION WITHOUT SPIN-BOSON COUPLING (IPR0)
% eta0 = 0;
% % build Hamiltonian
% Hsph = sparse(0);
% pol = sparse(0);
% Hn = sparse(0);
% for n=1:N
%     pol = pol + eta0*center_wf(n)*(1/sqrt(ener(n)/ener(1)))*(Oa_cell{n} - Oa_cell{n}');
%     Hn = Hn + ener(n)*On_cell{n};
% end
% %
% Hsph = (Omega/2)*Osp*expm(pol);
% Hsph = Hsph + Hsph';
% %
% H = Hsph + Hn  + (omegaz/2)*Osz;
% H = (H + H')/2;
% %
% % calculate the intitial density matrix
% %
% for n = 1:N
%     T(n) = ener(n)/log(1+1/n_val(n));
%     Z(n) = sum(exp(-ener(n)*[0:nc]/T(n)));
% end
% %
% rho0 = 1;
% %
% for n = 1:N
%    rho0 = (1/Z(n))*expm(-ener(n)*On_cell{n}/T(n))*rho0;
% end
% %
% rho0 = rho0*(Id + cos(theta)*Osz + sin(theta)*cos(phi)*Osx + sin(theta)*sin(phi)*Osy)/2;
% %
% % calculate 300 first eigenstates
% Neigens = min(500,length(H)-1);
% %
% [VVf,enerf0] = eigs(H,Neigens,meanE);
% %[VVf,enerf0] = eig(H);
% %
% rho_d = diag(VVf(:,:)'*rho0*VVf(:,:));
% IPR0 = 1/sum(rho_d.^2);
% ratio_IPR = IPR/IPR0;
%
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
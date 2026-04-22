function [P,nph,meansz,meansx,meansy,varsz,varsx,varsy,eta0] = sb_evolution_modes_Bloch_parallel(N,center,nc,nph,Omega,omegaz,axial,theta,phi,tmin,tmax,tstep)
%
% INPUT: 
%       N = number of ions
%       center = position of the spin in the chain (e.g. center = 1, spin at ion 1)
%       nc = cut-off of maximum number of phonons in the calculation
%
%       nph = vector with the mean phonon numbers for each vibrational
%       mode, e.g., nph = [0.1,0.2,0.3] -> mean values 0.1, 0.2, 0.3 for
%       axial vibrational modes n = 1,2,3, respectively
%
%       T = temperature (e.g., T = 100 with axial frequency 1, means <n> = 100)
%       Omega = Rabi frequency 
%       omegaz = detuning
%       axial = axial trapping frequency
%       theta, phi = angles in the Bloch sphere of the intial state:
%           \psi = cos(theta/2) |up> + sin(theta/2) e^(-i \phi) |down>
%       tmin, tmax, tstep:  minimum evolution time, maximum ev. time and time step
%       (timelimits are entered in units of 1/(2 \pi) microseconds)
%  OUTPUT
%       P(1,:) = values of time
%       P(2,:) = <sigma_z> values as a function of time
%       P(3,:) = <sigma_x> values as a function of time
%       P(4,:) = <sigma_y> values as a function of time
%       nph = number of phonons: nth column corresponds to nth vibrational mode
%                                nth file is the nth time step
%       meansz/x/y = time averaged value of <sigma^z/x/y>
%       varsz/x/y = variance of the flucutations in time of <sigma^z/x/y>
%                                   
    % calculate ion chain properties
    [Xmin,ener,center_wf,V] = cchain(N,center);
    % scale axial vibrational energies
    ener = ener*axial;
% define operators
[Oa_cell,On_cell,Oid_cell,Osz,Osm,Osp] = Operators(nc,N);
Osx = Osp + Osm;
Osy = -i*(Osp - Osm);
% define identity operator
Id = eye(length(On_cell{1}));
    % build vibrational Hamiltonian and spin-phonon Hamiltonian
    Hsph = 0;
    pol = 0;
    Hn = 0;
    eta0 = eta_calculator(25,200,axial);
    for n=1:N
        pol = pol + i*eta0*center_wf(n)*(1/sqrt(ener(n)/ener(1)))*(Oa_cell{n} + Oa_cell{n}');
        Hn = Hn + ener(n)*On_cell{n};
    end
    Hsph = (Omega/2)*Osp*expm(pol);
    Hsph = Hsph + Hsph';
    H = Hsph + Hn  + (omegaz/2)*Osz;
%
% build the initial thermal density matrix (|up> times thermal phonons)
% partition function
% AVOID T = 0
for n=1:N
    if nph(n) == 0
        nph(n) = 0.0001;
    end
end
%
for n = 1:N
    T(n) = ener(n)/log(1+1/nph(n));
    Z(n) = sum(exp(-ener(n)*[0:nc]/T(n)));
end
%
% build the initial thermal density matrix
%
rho0 = 1;
for n = 1:N
   rho0 = (1/Z(n))*expm(-ener(n)*On_cell{n}/T(n))*rho0;
end
% prepare the state in a pure initial spin state with angles (theta, phi)
% on the Bloch sphere
rho0 = rho0*(Id + cos(theta)*Osz + sin(theta)*cos(phi)*Osx + sin(theta)*sin(phi)*Osy)/2;
rho0 = rho0/trace(rho0);
count = 1;
%
% SOLVE TIME EVOLUTION (FULL HAMILTONIAN DIAGONALIZATION)
[VV,DD] = eig(H);
DD = diag(DD);
    % transform operators to the energy eigenbasis
    rho0 = VV'*rho0*VV;
    Osz = VV'*Osz*VV;
    Osx = VV'*Osx*VV;
    Osy = VV'*Osy*VV;
     for n=1:N
         On_cell{n} = VV'*On_cell{n}*VV;
     end
    %
P = zeros(4,108);    
parfor t = tmin:tstep:tmax
    U = diag(exp(-i*DD*t));
    rho = U*rho0*U';
    % conver time to microseconds
    P(1,count) = t/(2*pi);
    P(2,count) = real(trace(rho*Osz));
    P(3,count) = real(trace(rho*Osx));
    P(4,count) = real(trace(rho*Osy));
     %for n = 1:N
     %    nph(count,n) = real(trace(rho*On_cell{n}));
     %end
        count = count + 1;
%disp(t)
end
%
meansz = sum(P(2,:))/length(P(2,:));
varsz = sqrt(sum((P(2,:)-meansz).^2)/length(P(2,:)));
%
meansx = sum(P(3,:))/length(P(3,:));
varsx = sqrt(sum((P(3,:)-meansx).^2)/length(P(3,:)));
%
meansy = sum(P(4,:))/length(P(4,:));
varsy = sqrt(sum((P(4,:)-meansy).^2)/length(P(4,:)));
%
end
%
function [Oa_cell,On_cell,Oid_cell,Osz,Osm,Osp] = Operators(nc,Nsites)
%
n = 1;
j = n/2;
m_vec = -n/2+1:n/2;
sm = diag(sqrt(j*(j+1)-m_vec.*(m_vec-1)),-1); 
sp = sm';
sz = 2*diag(n/2:-1:-n/2);
%sx = (sm + sp);
%
%sy = i*(sm - sp);
a = diag(sqrt([1:nc]),1); 
%nn =  sparse(diag([0:nc]));
nn = a'*a;
is = eye(n+1);
ip = eye(nc+1);
Oa_cell = cell(Nsites,1);
On_cell = cell(Nsites,1);
Oid_cell = cell(Nsites,1);
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
Iph = eye(length(Oa_cell{1}));
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
%
%  FROM HERE ON: CODE TO CALCULATE PROPERTIES OF THE ION CHAIN: EQ.
%  POSITIONS, VIBRATIONAL ENERGIES AND PHONON WAVEFUNCTIONS
%
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

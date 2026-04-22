
function etatest(m,lam,wx);
% m atomic units
% lam nanometers
% wx MHz
hbar = 1.05457173*10^(-34);
mm = m*1.66*10^(-27);
wwx = 2*pi*wx*10^6;
k = 2*pi/(lam*10^(-9));
%
etatest = sqrt(hbar/(2*mm*wwx))*k
end
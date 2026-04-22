for omegaAx = 0.5:0.5:1;
N = 3;
nval = [0.5];
%filename = strcat('N',num2str(N),'Ax',num2str(10*omegaAx),'nval0',num2str(10*nval(1)),'.mat');
filename = strcat('N',num2str(N),'Ax',num2str(10*omegaAx),'nval',num2str(10*nval(1)),'.mat');
load(filename)
figure
ax1 = subplot(1,2,1)
tri1 = delaunay(sval(:,1),sval(:,2))
h1 = trisurf(tri1, sval(:,2), sval(:,1), ETHval(:,3));
set(ax1, 'CLim', [-0.1,0.9])
colorbar
hold on
map = colormap; % current colormap
for i=1:1:13
map(i,:) = [1,0,0];
end
colormap(ax1,map)
axis vis3d
xlabel('\Omega_{Rabi} (2\pi MHz)')
ylabel('\omega_z (2\pi MHz)')
title(strcat('\langle \sigma_z \rangle, ',num2str(N),' ions, \theta = 0, \omega_{axial} = 2\pi ',num2str(omegaAx),' MHz'))
ax2 = subplot(1,2,2)
tri2 = delaunay(varval(:,1),varval(:,2))
h2 = trisurf(tri2, varval(:,2), varval(:,1), varval(:,3));
set(ax2, 'CLim', [0.,1])
colorbar
hold on
map = colormap; % current colormap
for i=1:1:10
map(i,:) = [1,0,0];
end
colormap(ax2,map)
axis vis3d
xlabel('\Omega_{Rabi} (2\pi MHz)')
ylabel('\omega_z (2\pi MHz)')
title(strcat('Var\sigma_z ',num2str(N),' ions, \theta = 0, \omega_{axial} = 2\pi ',num2str(omegaAx),' MHz'))
end
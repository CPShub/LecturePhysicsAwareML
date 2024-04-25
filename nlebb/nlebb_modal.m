% -------------------------------------------------------------------------
% Physics-aware machine learning
% Cyber-Physical Simulation, TU Darmstadt
% -------------------------------------------------------------------------
% Nonlinear Euler-Bernoulli beam 
% Modal analysis 
% -------------------------------------------------------------------------

clear all;

% -------------------------------------------------------------------------
% --- INPUTS

% Geometric and material parameters
L = 0.2;            % Length [m]
W = 0.02;           % Width [m]
H = 0.02;           % Height [m]
E = 50e6;           % Young's modulus [Pa]
rho = 1100;         % Density [kg/m³]

% Cross-section parameters (assuming rectangular cross-section)
EA = E * W*H;       % E*A
EI = E * H^3*W/12;  % E*I
RA = rho * W*H;     % rho*A
param = [EA, EI, RA];

% Dirichlet boundary conditions (0:free, 1:roler, 2:simple, 3:clamped)
BC0 = 3;            % x=0
BC1 = 0;            % x=L

% Number of finite elements
ne = 4*2;      

% Visualization options
plotModes = 10;

% -------------------------------------------------------------------------
% --- DATA PREPARATION

% Mesh
nn = ne+1;                  % Number of nodes
XX = 0:(L/ne):L;            % Node positions
XE = [1:ne; 2:nn]';         % Node-to-element map
nu = 2*nn;                  % Number of shape functions
N = 2*nu;                   % Number of DOFs
UW = zeros(2,nu);           % Matrix of uw-values
UE = zeros(ne,8);           % DOF-to-element map
for i = 1:ne
    UE(i,:) = (4*i-4)+(1:8);
end

% Dirichlet DOFs
if (BC0 == 0)               % at x=0
    dofs_d = [];
elseif (BC0 == 1)
    dofs_d = 2; 
elseif (BC0 == 2)
    dofs_d = 1:2;
elseif (BC0 == 3)
    dofs_d = [1,2,4];
end
if (BC1 == 1)               % at x=L
    dofs_d = [dofs_d, 4*ne+2]; 
elseif (BC1 == 2)
    dofs_d = [dofs_d, 4*ne+(1:2)];
elseif (BC1 == 3)
    dofs_d = [dofs_d, 4*ne+[1,2,4]];
end
dofs_i = setdiff(1:N,dofs_d);   % Independent DOFs

% -------------------------------------------------------------------------
% --- MODAL ANALYSIS

% Initialize K and M
K = zeros(N,N);
M = zeros(N,N);

% Assembly loop over finite elements
for el = 1:ne
    
    % Data for evaluation  
    Xel = XX(:,XE(el,:));
    UWel = zeros(2,4);
    
    % Element evaluation
    [~, ~, Ke, Me] = nlebb_elem(Xel,UWel,param,@(x)[0,0]);
  
    % Assembly
    M(UE(el,:),UE(el,:)) = M(UE(el,:),UE(el,:)) + Me;
    K(UE(el,:),UE(el,:)) = K(UE(el,:),UE(el,:)) + Ke;
    
end

% Solve eigenvalue problem
Mii = M(dofs_i,dofs_i);
Kii = K(dofs_i,dofs_i);
[V,D] = eig(Kii,Mii);

dd = diag(D);
om = sqrt(dd);
tPer = 2*pi ./ om;

%save('eigU.mat','V');
%save('eigS.mat','om');

% Analytical eigenfrequencies (only for cantilever)
if (BC0==3 && BC1==0)
    
    omu = (1:2:21) * pi/2 * sqrt(E/rho) / L;
    bb = zeros(1,15);
    bb(1) = fzero(@(x)cosh(x*pi)*cos(x*pi)+1, [0 1]); 
    for k = 2:15
        bb(k) = fzero(@(x)cosh(x*pi)*cos(x*pi)+1, [bb(k-1)+0.1, bb(k-1)+1.5]); 
    end
    omw = (bb * pi / L).^2 * sqrt(EI/RA);
    oma = sort([omu, omw])';
    
    % Errors of eigenfrequencies
    omD = oma(1:20)-om(1:20);
    omR = omD ./ oma(1:20);
end

% -------------------------------------------------------------------------
% --- POST-PROCESSING

% Plot modes
if (plotModes)
    u = zeros(N,1);
    umodes = zeros(1,plotModes);
    umax = 0.2*L;
    wmax = 0.5*L;
    nlebbplots = nlebb_plot([],XX,XE,u,UE,5,0,1);
    for i = 1:plotModes
        u(dofs_i) = V(:,i);
        uim = max(abs(u(1:4:end)));
        wim = max(abs(u(2:4:end)));
        if (uim/umax > wim/wmax)
            u = u * umax/uim;
        else
            u = u * wmax/wim;
        end
        if (wim < 0.1)
            umodes(i) = 1;
        end
        if (u(6) < 0)
            u = -u;
        end
        nlebbplots = nlebb_plot(nlebbplots,XX,XE,u,UE,5,0,i+1);
    end 
end

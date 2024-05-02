% -------------------------------------------------------------------------
% Physics-aware machine learning
% Cyber-Physical Simulation, TU Darmstadt
% -------------------------------------------------------------------------
% Nonlinear Euler-Bernoulli beam 
% Static deformation example
% -------------------------------------------------------------------------

clear all;

% -------------------------------------------------------------------------
% --- INPUTS

% Geometric and material parameters
L = 0.20;           % Length [m]
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

% Axial & transversal line load [N/m]
load_f = 0;
load_q = 0; %-9.81*RA;
load = @(x) [load_f, load_q];               

% Point forces at points [1,2,3,4]*L/4 [N]
Nx = [0 0 0 0];
Qz = [0 0 0 20];  
My = 0; %-2;                % Moment at x=L [Nm]

% Number of finite elements
ne = 4*2;       

% Newton-Raphson parameters
rnMax = 20;     % Max. no. of iterations
eps = 1e-5;     % Tolerance for errors

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

% Neumann values
dofs_n = [(ne+1):ne:N, ((ne+1):ne:N)+1, N];
vals_n = [Nx, Qz, -My*ne/L]';


% -------------------------------------------------------------------------
% --- STATIC SOLVE WITH NEWTON-RAPHSON

% Initialization of DOF vector u
u = zeros(N,1);
nlebbplots = [];
femFlags = [1 1 1 0 1 1];

% Netwon-Raphson iterations
rn = 0;
ru = 1;
rr = 1;
while (rn < rnMax && ru > eps && rr > eps)

    rn = rn+1;

    % Finite element assembly
    [f, b, K, ~, Wint, Wext] = nlebb_assemble(XX, XE, u, UE, param, load, femFlags);
    
    % Point loads
    b(dofs_n) = b(dofs_n) + vals_n;
    Wext = Wext + dot(u(dofs_n),vals_n);

    % Residual and linear solve
    Ri = f(dofs_i) - b(dofs_i);
    Kii = K(dofs_i,dofs_i);
    Ui = Kii \ Ri;

    % Update
    u(dofs_i) = u(dofs_i) - Ui;

    % Errors
    ru = norm(Ui) / norm(u);
    rr = norm(Ri);

    % Print & plot
    fprintf("it=%2i, ru=%5.3e, rr=%5.3e\n", rn, ru, rr);
    nlebbplots = nlebb_plot(nlebbplots,XX,XE,u,UE,5,2,rn);

end

% Internal and external energy
fprintf('Solution:   Wint=%5.3e, Wext=%5.3e, dW=%5.3e\n', ...
    Wint, Wext, Wext-Wint);

% Analytical solution linear cantilever (only for load_q, Qz & Mp)
wana = @(x)  1/EI/24 * x.^2 .* (Qz(4)*4*(3*L - x) + load_q*(6*L^2 - 4*L*x + x.^2) - My*12);
wana1 = @(x) 1/EI/6 * x .* (Qz(4)*3*(2*L - x) + load_q*(3*L^2 - 3*L*x + x.^2) - My*6);
wana2 = @(x) 1/EI/2 * (Qz(4)*2*(L - x) + load_q*(L^2 - 2*L*x + x.^2) - My*2);
uana0 = @(x) -(Qz(4)/EI)^2 / 8 * x.^3 .* (4/3*L^2 - L*x + 1/5*x.^2);
uana1 = @(x) -(Qz(4)/EI)^2 / 8 * x.^2 .* (2*L-x).^2;
uana2 = @(x) -(Qz(4)/EI)^2 / 2 * x .* (L-x).*(2*L-x);

[qp,qw] = gauss1d(7,0,L);
Wextana = sum(load_q * wana(qp) .* qw) + Qz(4)*wana(L) - My*wana1(L);
Wintana = sum(0.5*EI * wana2(qp).^2 .* qw);
dWana = Wextana-Wintana;

fprintf('Analytical: Wint=%5.3e, Wext=%5.3e, dW=%5.3e\n', ...
    Wintana, Wextana, dWana);

xxx = 0:(0.01*L):L;
plot(nlebbplots.Children(3), xxx, wana(xxx), 'Displayname', 'w ana.');
plot(nlebbplots.Children(3), xxx, uana0(xxx), '--', 'Displayname', 'u ana.');
plot(nlebbplots.Children(2), xxx, wana1(xxx), 'Displayname', 'w` ana.');
plot(nlebbplots.Children(2), xxx, uana1(xxx), '--', 'Displayname', 'u` ana.');
plot(nlebbplots.Children(1), xxx, wana2(xxx), 'Displayname', 'w`` ana.');
plot(nlebbplots.Children(1), xxx, uana2(xxx), '--', 'Displayname', 'u`` ana.');

return



% -------------------------------------------------------------------------
% Physics-aware machine learning
% Cyber-Physical Simulation, TU Darmstadt
% -------------------------------------------------------------------------
% Nonlinear Euler-Bernoulli beam 
% Dynamic deformation example with explicit time integration
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

% Axial & transversal line load [N/m]
load_f = 0;
load_q = 0; % -0.981*RA*0.1;
load = @(x,t) [load_f, load_q];    

% Point forces at points [1,2,3,4]*L/4 [N]
tPer = .5;                  % Vibration period [s]
Nx = @(t) [0 0 0 0];
Qz = @(t) [0 0 0 -40*sin(2*pi/tPer*t)];  
%Qz = @(t) [0 0 0 -10*(1+t)*sin(2*pi/tPer*t)];  
My = @(t) 0;                % Moment at x=L [Nm]

% Time discretization parameters
tend = 1*tPer;              % End time
stepsPer = 8 * 40;          % Time steps per period
dt = tPer/stepsPer;         % Time step size
twrite = stepsPer / 16 *25;     % Write to command line every ... time steps

% Number of finite elements
ne = 4*2;       

% Visualization options
plotBeamSteps = stepsPer / 8;   % Plot deformed beam every ... time steps (0: no plot)
plotBeamDerivs = 0;             % Plot also 1st or 2nd derivatives
plotTimePts = [1+ne/2 1+ne];    % Plot deformation over time at node points ([]: no plot)
plotEnergy = 0;                 % Plot energies over time
plotMovieSteps = stepsPer / 16 * 0;  % Create a movie frame every ... time steps (0: no movie)

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
Ni = length(dofs_i);

% Neumann values
dofs_n = [(ne+1):ne:N, ((ne+1):ne:N)+1, N];
vals_n = @(t) [Nx(t), Qz(t), -My(t)*ne/L]';

% -------------------------------------------------------------------------
% --- TIME INTEGRATION WITH IMPLICIT NEWMARK METHOD

ti = 1;

% Initialization of DOF vector u,u',u", mass matrix M
U0 = zeros(N,1);    % Initial condition U=0
U1 = zeros(N,1);    % Initial condition U'=0

% Initialize plotting
if (plotBeamSteps > 0)
    nlebbplots = nlebb_plot([],XX,XE,U0,UE,5,plotBeamDerivs,ti);
end

% Pre-compute mass matrix
femFlags = [0 0 0 1 0 0];
[~, ~, ~, M] = nlebb_assemble(XX, XE, U0, UE, param, @(x)load(x,t), femFlags);
Mii = sparse(M(dofs_i,dofs_i));

% Call MATLAB integrator
odeOptions = odeset('RelTol',1e-1,'Stats','on');
Y0 = zeros(2*Ni,1);
tic
[tt,Yall] = ode23(@(t,Y) odefun(t,Y,XX,XE,UE,param,load,dofs_i,dofs_n,vals_n,Mii), ...
    [0,tend], Y0, odeOptions);
toc

tsteps = length(tt);
U0all = zeros(N,tsteps);
U1all = zeros(N,tsteps);
U0all(dofs_i,:) = Yall(:,1:Ni)';
U1all(dofs_i,:) = Yall(:,Ni+(1:Ni))';

% -------------------------------------------------------------------------
% --- POST-PROCESSING

% Plot displacements over time
if (~isempty(plotTimePts))
    figure; 
    set(gcf,'Color','white');
    ax1 = subplot(1,2,1); hold on;
    title('u');
    grid on;
    ax2 = subplot(1,2,2); hold on;
    title('w');
    grid on;
    
    for i=1:length(plotTimePts)
        plot(ax1, tt, U0all(4*plotTimePts(i)-3,:), ...
            'Displayname',sprintf('x=%4.2f', XX(plotTimePts(i))));
        plot(ax2, tt, U0all(4*plotTimePts(i)-2,:), ...
            'Displayname',sprintf('x=%4.2f', XX(plotTimePts(i))));
    end
    legend;
end

% Plot energies over time
if (plotEnergy)
    figure; 
    hold on;
    set(gcf,'Color','white');
    title('energies');
    grid on;
    plot(tt, Wall(1,:), 'Displayname', 'Kinetic energy');
    plot(tt, Wall(2,:), 'Displayname', 'Internal energy');
    plot(tt, Wall(3,:), 'Displayname', 'External work');
    legend;
end

% Create, show (and save) movie
if (plotMovieSteps > 0)

    % Determine size of frame window
    u0min = min(min(U0all(1,:))-0.02*L,-0.2*L);
    u0min = floor(u0min * 10^(-floor(log10(L)))) * 10^floor(log10(L));
    u1max = max(max(U0all(N-3,:))+0.02*L, 0.2*L);
    u1max = ceil(u1max * 10^(-floor(log10(L)))) * 10^floor(log10(L));
    Wmin = min(min(min(U0all(2:4:N,:)))-0.05*L,-0.2*L);    
    Wmin = floor(Wmin * 10^(-floor(log10(L)))) * 10^floor(log10(L));
    Wmax = max(max(max(U0all(2:4:N,:)))+0.05*L, 0.2*L);
    Wmax = ceil(Wmax * 10^(-floor(log10(L)))) * 10^floor(log10(L));

    % Create frames
    nlebbframe = []; 
    frames(ceil(tend/dt/plotMovieSteps)+1) = struct('cdata',[],'colormap',[]);
    for ti = 1:plotMovieSteps:tsteps
    
        nlebbframe = nlebb_frame(nlebbframe,XX,XE,U0all(:,ti),UE,tt(ti),5);
        axis equal; axis([u0min L+u1max Wmin Wmax]); 
        frames(floor(ti/plotMovieSteps)+1) = getframe(gcf);
    end   

    % Make movie
    movie(nlebbframe,frames,1);

    % Save movie
    if (0)
        vwriter = VideoWriter("cantilever.avi");
        open(vwriter);
        writeVideo(vwriter,frames);
        close(vwriter);
    end
end


return

% -------------------------------------------------------------------------
% --- FUNCTION FOR MATLAB INTEGRATORS

function dY = odefun(t, Y, XX, XE, UE, param, load, dofs_i, dofs_n, vals_n, Mii)

N = 4*length(XX);
Ni = length(dofs_i);
U0 = zeros(N,1);
U0(dofs_i) = Y(1:Ni);
U1i = Y((Ni+1):end);

femFlags = [1 1 0 0 0 0];
[f, b] = nlebb_assemble(XX, XE, U0, UE, param, @(x)load(x,t), femFlags);
b(dofs_n) = b(dofs_n) + vals_n(t);
%Mii = sparse(M(dofs_i,dofs_i));
%Mid = sparse(diag(sum(M(dofs_i,dofs_i))));

%dY = [U1i; M(dofs_i,dofs_i) \ (b(dofs_i) - f(dofs_i))];
dY = [U1i; Mii \ (b(dofs_i) - f(dofs_i))];

end



% -------------------------------------------------------------------------
% Physics-aware machine learning
% Cyber-Physical Simulation, TU Darmstadt
% -------------------------------------------------------------------------
% Nonlinear Euler-Bernoulli beam 
% Dynamic deformation example with implicit Newmark time integration
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
%load = @(x,t) [0, -0.981*RA];       
load = @(x,t) [0, 0];  

% Point forces at points [1,2,3,4]*L/4 [N]
tPer = 1;           % Vibration period [s]
Fp = @(t) [0 0 0 0, ...     % x-dir.
           0 0 0 -1*sin(2*pi/tPer*t)  ];      % z-dir.
%Fp = @(t) [0 0 0 0, ...                     % x-dir.
%           0 0 0 -10*sin(2*pi/tPer*t)];     % z-dir.
My = @(t) 0;                 % Moment at x=L

% Time discretization parameters
tend = 5*tPer;              % End time
stepsPer = 8 * 40;          % Time steps per period
dt = tPer/stepsPer;         % Time step size
twrite = stepsPer / 16;     % Write to command line every ... time steps

% Number of finite elements
ne = 5*2;       

% Visualization options
plotBeamSteps = stepsPer / 8;   % Plot deformed beam every ... time steps (0: no plot)
plotBeamDerivs = 1;             % Plot also 1st or 2nd derivatives
plotTimePts = [1+ne/2 1+ne];    % Plot deformation over time at node points ([]: no plot)
plotEnergy = 1;                 % Plot energies over time
plotMovieSteps = stepsPer / 16 * 1;  % Create a movie frame every ... time steps (0: no movie)

% Newton-Raphson parameters
rnMax = 20;     % Max. no. of iterations
eps = 1e-5;     % Tolerance for errors

% Newmark parameters
gamma = 0.6;        
beta = gamma / 2.; 

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
vals_n = @(t) [Fp(t), -My(t)*ne/L]';

% -------------------------------------------------------------------------
% --- TIME INTEGRATION WITH IMPLICIT NEWMARK METHOD

% Initalization of time variables
t = 0;
ti = 1;
tsteps = round(tend/dt+1);
tt = zeros(tsteps,1);

% Initialization of DOF vector u,u',u", mass matrix M
U0 = zeros(N,1);    % Initial condition U=0
U1 = zeros(N,1);    % Initial condition U'=0
U2 = zeros(N,1);
M = zeros(N,N);

% Arrays for saving values of u,u',u",f,b for all time steps
U0all = zeros(N,tsteps);
U1all = U0all;
U2all = U0all;
Fall = U0all;
Ball = U0all;
Wall = zeros(3,tsteps);     % Kinetic, internal & external energy
dWall = zeros(3,tsteps);     % Rates of kinetic, internal & external energy

% Initialize plotting
if (plotBeamSteps > 0)
    nlebbplots = nlebb_plot([],XX,XE,U0,UE,5,plotBeamDerivs,ti);
end

% Loop over time steps
tic
femFlags = [1 1 1 1 1 1];
while (t < tend)

    % Increment time
    ti = ti+1;
    t = t+dt;
    tt(ti) = t;

    % History vectors
    bOld = (M*U0) / (beta*dt^2) + (M*U1) / (beta*dt) + (M*U2) * (1-2*beta)/(2*beta);
    U00 = U0;
    U20 = U2;

    % Netwon-Raphson iterations
    rn = 0;
    ru = 1;
    rr = 1;
    while (rn < rnMax && (ru > eps || rr > eps))
    
        rn = rn+1;

        % % Initialize for current iteration
        % K = zeros(N,N);
        % f = zeros(N,1);
        % b = f;
        % 
        % % Assembly loop over finite elements
        % for el = 1:ne
        % 
        %     % Data for evaluation  
        %     Xel = XX(:,XE(el,:));
        %     Uel = U0(UE(el,:));
        %     UWel = [Uel(1:2:7)'; Uel(2:2:8)'];
        % 
        %     % Element evaluation
        %     if (ti == 2 && rn == 1)
        %         [fe, be, Ke, Me] = nlebb_elem(Xel,UWel,param,@(x)load(x,t));
        %         M(UE(el,:),UE(el,:)) = M(UE(el,:),UE(el,:)) + Me;
        %     else
        %         [fe, be, Ke] = nlebb_elem(Xel,UWel,param,@(x)load(x,t));
        %     end
        % 
        %     % Assembly
        %     K(UE(el,:),UE(el,:)) = K(UE(el,:),UE(el,:)) + Ke;
        %     f(UE(el,:)) = f(UE(el,:)) + fe;
        %     b(UE(el,:)) = b(UE(el,:)) + be;
        % 
        % end

        % Finite element assembly
        if (ti == 2 && rn == 1)
            [f, b, K, M, Wint, Wext] = nlebb_assemble(XX, XE, U0, UE, param, load, femFlags);
            femFlags = [1 1 1 0 1 1];
        else
            [f, b, K, ~, Wint, Wext] = nlebb_assemble(XX, XE, U0, UE, param, load, femFlags);
        end        

        % Point loads
        b(dofs_n) = b(dofs_n) + vals_n(t);
        Wext = Wext + dot(U0(dofs_n),vals_n(t));
        
        % Residual and linear solve
        R = (M*U0) / (beta*dt^2) + f - b - bOld;
        Ri = R(dofs_i);
        KT = M / (beta*dt^2) + K;
        Kii = KT(dofs_i,dofs_i);
        Ui = Kii \ Ri;

        % Update
        U0(dofs_i) = U0(dofs_i) - Ui;
    
        % Errors
        ru = norm(Ui) / norm(U0);
        rr = norm(Ri);
    
    end

    % Convergence check
    if (rn >= rnMax)
        fpvals = vals_n(t);
        fprintf("t=%5.2f, rn=%2i, ru=%5.3e, rr=%5.3e, fp=%6.3f\n", ...
            t, rn, ru, rr, fpvals(8));
        fprintf("Max. no of Newton iterations exceeded - abort\n");
        t = tend + 1;
        continue
    end

    % Update velocities and accelerations
    U2 = (U0-U00)/(beta*dt^2) - U1/(beta*dt) - (1-2*beta)/(2*beta)*U2;
    U1 = U1 + dt*(1-gamma)*U20 + dt*gamma*U2;

    % Save vectors
    U0all(:,ti) = U0;
    U1all(:,ti) = U1;
    U2all(:,ti) = U2;
    Fall(:,ti) = f;
    Ball(:,ti) = b;

    % Compute energies
    Wkin = 0.5*dot(U1,M*U1);
    %Wint = 0.5*dot(f,U0);
    %Wext = dot(b,U0);
    Wall(:,ti) = [Wkin; Wint; Wext];

    dWall(:,ti) = [dot(U2,M*U1); dot(f,U1); dot(b,U1)];

    % Print
    if (rem(ti,twrite) == 1)
        fpvals = vals_n(t);
        fprintf("t=%5.2f, rn=%2i, ru=%5.3e, rr=%5.3e, fp=%6.3f\n", ...
            t, rn, ru, rr, fpvals(8));
        fprintf('         Wkin=%5.3e, Wint=%5.3e, Wext=%5.3e\n', ...
            Wkin, Wint, Wext);
    end

    % Plots
    if (rem(ti,plotBeamSteps) == 1)
        nlebbplots = nlebb_plot(nlebbplots,XX,XE,U0,UE,5,plotBeamDerivs,ti);
    end

end

toc

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

% Write data to files
if (0)
    writematrix(tt,'dataT.dat','Delimiter','\t');
    writematrix(U0all,'dataU0.dat','Delimiter','\t');
    writematrix(Fall,'dataF.dat','Delimiter','\t');
end

return



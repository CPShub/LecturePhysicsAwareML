% -------------------------------------------------------------------------
% Physics-aware machine learning
% Cyber-Physical Simulation, TU Darmstadt
% -------------------------------------------------------------------------
% Nonlinear Euler-Bernoulli beam 
% Dynamic deformation example with implicit Newmark time integration
% using projection-based MOR with eigenmodes or POD and DEIM
% -------------------------------------------------------------------------

clear all;

% -------------------------------------------------------------------------
% --- INPUTS

% Projection-based MOR
qmode = 2;          % 0: Off, 1: Modal, 2: POD/SVD
qm = 4;             % Number of modes for projection
qdeim = 0; % 12;    % Use DEIM for f with ... modes

% Neural network-based approximation
qnn = 0;            % 0: Off, 1: No energy, 2: Energy
fnw = fullfile('..', 'FFNN_ROM', 'data', 'weights.txt');

% Axial & transversal line load [N/m]
load_f = 0;
load_q = 0; % -0.981*RA*0.1;
load = @(x,t) [load_f, load_q];    

% Point forces at points [1,2,3,4]*L/4 [N]
tPer = .01;                  % Vibration period [s]
Nx = @(t) [0 0 0 0];
Qz = @(t) [0 0 0 -10*sin(2*pi/tPer*t)];  
% Qz = @(t) [0 0 0 -10*(1+t)*sin(2*pi/tPer*t)];
% Qz = @(t) [0 0 0 multiphase_multisin(2,0.085,100,1,3,t)]; % Multisine train 1
% Qz = @(t) [0 0 0 multiphase_multisin(2,0.085,100,2,3,t)]; % Multisine train 2
% Qz = @(t) [0 0 0 multiphase_multisin(2,0.085,100,3,3,t)]; % Multisine train 3
% Qz = @(t) [0 0 0 multiphase_multisin(1,0.103,78,1,1,t)];  % Multisine test 1
% Qz = @(t) [0 0 0 1.5*sin(2*pi*6*t)];                      % Sine test
% Qz = @(t) [0 0 0 -10*(t>.2)];                            % Step test
% Qz = @(t) [0 0 0 -10*(t>.5)*(t<.51)];                   % Dirac test
% Qz = @(t) [0 0 0 -2*t];                                   % Quasi static
My = @(t) 0;                % Moment at x=L [Nm]


% -------------------------------------------------------------------------
% --- CALL FUNCTION FOR DATA GENERATION (BEAM SIMULATION)

[q0all, Qfall, QKQall] = nlebb_dynamic_fun(load, Nx, Qz, My, tPer, qmode, qm, qdeim, qnn, fnw);

% -------------------------------------------------------------------------
% --- SAVE REDUCED DOFS AND INTERNAL FORCE VECTORS

writematrix(q0all,'q0all.txt')
writematrix(Qfall,'Qfall.txt')
writematrix(QKQall,'QKQall.txt')

% -------------------------------------------------------------------------
% -------------------------------------------------------------------------

function [q0all, Qfall, QKQall] = nlebb_dynamic_fun(load, Nx, Qz, My, tPer, qmode, qm, qdeim, qnn, fnw)

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
%-- input 

% Point forces at points [1,2,3,4]*L/4 [N]
%-- input

% Time discretization parameters
tend = 5*tPer;              % End time
stepsPer = 8 * 20 * 2;          % Time steps per period
dt = tPer/stepsPer;         % Time step size
twrite = stepsPer / 16;     % Write to command line every ... time steps

% Number of finite elements
ne = 4*2;       

% Visualization options
plotBeamSteps = stepsPer / 8;   % Plot deformed beam every ... time steps (0: no plot)
plotBeamDerivs = 0;             % Plot also 1st or 2nd derivatives
plotTimePts = [1+ne/2 1+ne];    % Plot deformation over time at node points ([]: no plot)
plotEnergy = 1;                 % Plot energies over time
plotMovieSteps = stepsPer / 16 * 0;  % Create a movie frame every ... time steps (0: no movie)

% Newton-Raphson parameters
rnMax = 20;     % Max. no. of iterations
eps = 1e-5;     % Tolerance for errors

% Newmark parameters
gamma = 0.6;        
beta = gamma / 2.; 

plotModes = 1;

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
dofs_i = setdiff(1:N,dofs_d); % Independent DOFs
Ni = length(dofs_i);

% Neumann values
dofs_n = [(ne+1):ne:N, ((ne+1):ne:N)+1, N];
vals_n = @(t) [Nx(t), Qz(t), -My(t)*ne/L]';

% -------------------------------------------------------------------------
% --- LOAD MODES AND PREPARE PROJECTION

% Load projection matrix
if (qmode == 0)
    qm = Ni;
    loadQ = eye(Ni,Ni);
elseif (qmode == 1)
    loadS = importdata("eigS.mat");
    loadQ = importdata("eigU.mat");
elseif (qmode == 2)
    loadS = importdata("svdUS1.mat");
    loadQ = importdata("svdUU1.mat");
end
if (size(loadQ,1) ~= Ni)
    if (size(loadQ,1) == N)
        loadQ = loadQ(dofs_i,:);
    else
        error("Wrong size of Q!");
    end
end
Q = loadQ(:,1:qm);

% Plot modes
if (qmode && plotModes)
    u = zeros(N,1);
    nlebbplots = nlebb_plot([],XX,XE,u,UE,5,0,i);
    for i = 1:qm
        u(dofs_i) = Q(:,i);
        nlebbplots = nlebb_plot(nlebbplots,XX,XE,u,UE,5,0,i+1);
    end 
end


% Use DEIM for f
if (qdeim)
    loadFS = importdata("svdFSi1.mat");
    loadFU = importdata("svdFUi1.mat");
    %loadFS = importdata("svdFSi.mat");
    %loadFU = importdata("svdFUi.mat");
    if (size(loadFU,1) ~= Ni)
        if (size(loadFU,1) == N)
            loadFU = loadFU(dofs_i,:);
        else
            error("Wrong size of FU!");
        end
    end
    F0U = loadFU(:,1:qdeim);
    F0P = deim(F0U);
    Qd = (F0U(F0P, :)' \ (F0U'*Q))';
end

% Use NN for f
if (qnn==1)
    units = [16 16 16 16 qm]; % Number of nodes in hidden and output layers
    activations = {@softplus @softplus @softplus @softplus @linear};
    [Wnn,bnn] = read_weights(units,fnw);
elseif (qnn==2)
    units = [16 16 1]; % Number of nodes in hidden and output layers
    activations = {@softplus @softplus @linear};
    [Wnn,bnn] = read_weights(units,fnw);
end

% -------------------------------------------------------------------------
% --- TIME INTEGRATION WITH IMPLICIT NEWMARK METHOD - WITH MOR

% Initalization of time variables
t = 0;
ti = 1;
tsteps = round(tend/dt+1);
tt = zeros(tsteps,1);

% Initialization of DOF vector u 
U0 = zeros(N,1);
M = zeros(N,N);

% Initialization of reduced DOF vector q
q0 = zeros(qm,1);
q1 = zeros(qm,1);
q2 = zeros(qm,1);
QMQ = 0;
isInit = 0;

% Arrays for saving values of u,q,q',q" for all time steps
U0all = zeros(N,tsteps);
q0all = zeros(qm,tsteps);
q1all = q0all;
q2all = q0all;
Qfall = q0all;
QKQall = zeros(qm^2, tsteps);
Wall = zeros(3,tsteps);     % Kinetic, internal & external energy

% Initialize plotting
if (plotBeamSteps > 0)
    nlebbplots = nlebb_plot([],XX,XE,U0,UE,5,plotBeamDerivs,ti);
end

% Loop over time steps
while (t < tend)

    % Increment time
    ti = ti+1;
    t = t+dt;
    tt(ti) = t;    

    % History vectors
    QbOld = (QMQ*q0) / (beta*dt^2) + (QMQ*q1) / (beta*dt) + (QMQ*q2) * (1-2*beta)/(2*beta);
    q00 = q0;
    q20 = q2;

    % Netwon-Raphson iterations
    lastwarn('', '');
    rn = 0;
    ru = 1;
    rr = 1;
    while (rn < rnMax && (ru > eps || rr > eps))

        rn = rn+1;

        % Initialize for current iteration
        K = zeros(N,N);
        f = zeros(N,1);
        b = f;

        % Assembly
        for el = 1:ne

            % Data for evaluation  
            Xel = XX(:,XE(el,:));
            Uel = U0(UE(el,:));
            UWel = [Uel(1:2:7)'; Uel(2:2:8)'];

            % Element evaluation
            if (isInit)
                [fe, be, Ke] = nlebb_elem(Xel,UWel,param,@(x)load(x,t));
            else
                [fe, be, Ke, Me] = nlebb_elem(Xel,UWel,param,@(x)load(x,t));
                M(UE(el,:),UE(el,:)) = M(UE(el,:),UE(el,:)) + Me;
            end

            % Assembly
            K(UE(el,:),UE(el,:)) = K(UE(el,:),UE(el,:)) + Ke;
            f(UE(el,:)) = f(UE(el,:)) + fe;
            b(UE(el,:)) = b(UE(el,:)) + be;

        end

        % Point loads
        b(dofs_n) = b(dofs_n) + vals_n(t);

        % Project
        if (~isInit)
            QMQ = Q'*M(dofs_i,dofs_i)*Q;
            isInit = 1;
        end

        fi = f(dofs_i);
        Kii = K(dofs_i,dofs_i);
        if (qdeim)
            Qf = Qd * fi(F0P);
            QKQ = Qd * Kii(F0P,:) * Q;     
            % Note: An efficient assembly for DEIM should only sample the
            %       required rows of fi(F0P)! Assembling the whole f is
            %       completely inefficient and only done here for demo.
        elseif (qnn==1)
            [Qf,QKQ] = mlp(Wnn,bnn,activations,q0);
            % Note: An efficient use of the neural network would mean
            %       disabling the assembly of f and K, which was not done
            %       here to the changes less intrusive to the code.
        elseif (qnn==2)
            [~,Qf,QKQ] = mlp(Wnn,bnn,activations,q0);
            Qf = Qf';
            QKQ = reshape(QKQ,qm,qm);
            % Note: An efficient use of the neural network would mean
            %       disabling the assembly of f and K, which was not done
            %       here to the changes less intrusive to the code.
        else
            Qf = Q'*fi;
            QKQ = Q'*Kii*Q;
        end
        Qb = Q'*b(dofs_i);        

        % Residual and linear solve
        R = (QMQ*q0) / (beta*dt^2) + Qf - Qb - QbOld;
        KT = QMQ / (beta*dt^2) + QKQ;
        dq = KT \ R;

        if (lastwarn())
            problem = 1;
        end
        
        % Update
        q0 = q0 - dq;
        U0(dofs_i) = Q*q0;

        % Errors
        ru = norm(dq) / norm(q0);
        rr = norm(R);

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
    q2 = (q0-q00)/(beta*dt^2) - q1/(beta*dt) - (1-2*beta)/(2*beta)*q2;
    q1 = q1 + dt*(1-gamma)*q20 + dt*gamma*q2;

    % Save vectors
    q0all(:,ti) = q0;
    q1all(:,ti) = q1;
    q2all(:,ti) = q2;
    Qfall(:,ti) = Qf;
    U0all(:,ti) = U0;
    Qfall(:,ti) = Qf;
    QKQall(:,ti) = reshape(QKQ, 1, []);

    % Compute energies
    Wkin = 0.5*dot(q1,QMQ*q1);
    Wint = 0.5*dot(q0,Qf);
    Wext = dot(q0,Qb);
    Wall(:,ti) = [Wkin, Wint, Wext]';

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

% -------------------------------------------------------------------------
% --- POST-PROCESSING

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

% Plot mode amplitudes
if (plotModes)
    figure; hold on;
    title('Mode amplitudes q_i');
    for i = 1:qm
        plot(tt, q0all(i,:), 'Displayname', sprintf('q_%i', i));
    end
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

% -------------------------------------------------------------------------
% --- NEURAL NETWORK ACTIVATION FUNCTIONS
function [y,dy,ddy] = softplus(x)
    y = softplus(x);
    if nargout > 1
        dy = sigmoid(x);
        if nargout > 2
            ddy = dsigmoid(dy);
        end
    end
    
    function y = softplus(x)
        y = log(1 + exp(x));
    end
    function y = sigmoid(x)
        y = 1 ./ (1 + exp(-x));
    end
    function y = dsigmoid(x)
        y = x.*(1 - x);
    end
end

function [y,dy,ddy] = linear(x)
    y = x;
    if nargout > 1
        dy = x./x;
        if nargout > 2
            ddy = x * 0;
        end
    end
end

end




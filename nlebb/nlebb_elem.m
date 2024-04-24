% -------------------------------------------------------------------------
% Physics-aware machine learning
% Cyber-Physical Simulation, TU Darmstadt
% -------------------------------------------------------------------------
% Nonlinear Euler-Bernoulli beam 
% Element vectors and matrices evaluation with cubic Hermite splines
% -------------------------------------------------------------------------

function [Fe, Be, Ke, Me, Wie, Wee] = nlebb_elem(Xe, Ue, param, load, flags)

% Dimensions
d = 1;                  % Geometry dimension
dd = 2;                 % Solution dimension
ne = 4;                 % Number of nodes

% Check inputs
if (size(Xe,1) ~= d || size(Xe,2) ~= 2)
    error("nlebb_elem: wrong size of Xe"); 
end
if (size(Ue,1) ~= dd || size(Ue,2) ~= ne)
    error("nlebb_elem: wrong size of Ue"); 
end
if (length(param) ~= 3)
    error("nlebb_elem: wrong size of param"); 
end
if (nargin < 4)
    load = @(x)[0, 0];
end
if (nargin < 5)
    flags = ones(6,1);
end

% Initalization
EA = param(1);          % Axial stiffness (E*A)
EI = param(2);          % Bending stiffness (E*I)
rA = param(3);          % Line density (rho*A)
Ne = dd*ne;             % Number of DOFs
Ke = zeros(Ne,Ne);      % Element stiffness matrix
Kek = Ke;
Me = Ke;                % Element mass matrix
Mek = Ke;
Fe = zeros(Ne,1);       % Element internal force vector
Fek = Fe;
Be = zeros(Ne,1);       % Element external force vector
Wie = 0;                % Element internal energy       
Wee = 0;                % Element external work
Ae = 0;                 % Element area
dofU = 1:2:7;           % u-DOFs
dofW = 2:2:8;           % w-DOFs

% Quadrature points
qn = 4;
[qp, qw] = gauss1d(qn,0,1);

% Integration loop
for k = 1:qn

    % Current coordinates Xi and X
    Xi = qp(k);
    XX = Xe(1) + Xi * (Xe(2)-Xe(1));
    
    % Evaluation of shape functions and parametric gradients
    H0 = [1-3*Xi^2+2*Xi^3, Xi-2*Xi^2+Xi^3, 3*Xi^2-2*Xi^3, -Xi^2+Xi^3];  
    H1 = [-6*Xi+6*Xi^2, 1-4*Xi+3*Xi^2, 6*Xi-6*Xi^2, -2*Xi+3*Xi^2];
    H2 = [-6+12*Xi, -4+6*Xi, 6-12*Xi, -2+6*Xi];

    % Jacobian (parametric gradient of coordinate transformation)
    J = abs(Xe(2)-Xe(1));
    qwJ = J * qw(k);
    H1 = H1 / J;
    H2 = H2 / J^2;

    % Evaluation of u', w', w"
    u1 = dot(H1, Ue(1,:));
    w1 = dot(H1, Ue(2,:));
    w2 = dot(H2, Ue(2,:));

    if (flags(2) || flags(6))
        lv = load(XX);
    end
    
    % Integrand evaluation for element force vector
    if (flags(1))
        Fek(dofU) = EA * (u1 + 0.5 * w1^2) * H1';
        Fek(dofW) = (EA * (w1 * u1 + 0.5 * w1^3)) * H1' ...
            + (EI * w2) * H2';
        Fe = Fe + qwJ * Fek;
    end
    if (flags(2))
        Be(dofU) = Be(dofU) + qwJ * lv(1) * H0';
        Be(dofW) = Be(dofW) + qwJ * lv(2) * H0';
    end

    % Integrand evaluation for element stiffness matrix
    if (flags(3) && nargout > 2)  
        H1H1 = (H1' * H1) * EA;
        Kek(dofU,dofU) = H1H1;
        Kek(dofU,dofW) = w1 * H1H1;
        Kek(dofW,dofU) = w1 * H1H1;
        Kek(dofW,dofW) = (u1 + 1.5 * w1^2) * H1H1 + EI * (H2' * H2);
        Ke = Ke + qwJ * Kek;
    end
    
    % Integrand evaluation for element stiffness matrix
    if (flags(4) && nargout > 3)   
        Mek(dofU,dofU) = rA * (H0' * H0);
        Mek(dofW,dofW) = Mek(dofU,dofU);
        Me = Me + qwJ * Mek;
    end 

    % Integrand evaluation for element internal energy & external work
    if (flags(5) && nargout > 4)   
        Wek = 0.5 * (EA * (u1 + 0.5*w1^2)^2 + EI * w2^2);
        Wie = Wie + qwJ * Wek;
    end 
    if (flags(6) && nargout > 5)   
        u0 = dot(H0, Ue(1,:));
        w0 = dot(H0, Ue(2,:));
        Wek = u0 * lv(1) + w0 * lv(2);
        Wee = Wee + qwJ * Wek;
    end 

    % Area - for reference
    Ae = Ae + qwJ;
end

end


% -------------------------------------------------------------------------
% Physics-aware machine learning
% Cyber-Physical Simulation, TU Darmstadt
% -------------------------------------------------------------------------
% Nonlinear Euler-Bernoulli beam 
% Assembly of global force vectors and matrices
% -------------------------------------------------------------------------

function [f, b, K, M, Wint, Wext] = nlebb_assemble(XX, XE, U, UE, param, load, flags)

% Dimensions
N = length(U);
ne = size(XE, 1);

% Check inputs
if (length(XX) ~= N/4)
    error("nlebb_assemble: wrong size of XX"); 
end
if (size(XE,2) ~= 2)
    error("nlebb_assemble: wrong size of XE"); 
end
if (size(UE,1) ~= ne || size(UE,2) ~= 8)
    error("nlebb_assemble: wrong size of UE"); 
end
if (length(param) ~= 3)
    error("nlebb_assemble: wrong size of param"); 
end
if (nargin < 6)
    load = @(x)[0, 0];
end
if (nargin < 7)
    flags = ones(6,1);
end

% Initialize 
if (flags(1)), f = zeros(N,1); else, f = []; end
if (flags(2)), b = zeros(N,1); else, b = []; end
if (flags(3)), K = zeros(N,N); else, K = []; end
if (flags(4)), M = zeros(N,N); else, M = []; end
Wint = 0;
Wext = 0;

% Assembly loop over finite elements
for el = 1:ne
    
    % Data for evaluation  
    Xel = XX(:,XE(el,:));
    Uel = U(UE(el,:));
    UWel = [Uel(1:2:7)'; Uel(2:2:8)'];
    
    % Element evaluation
    [fe, be, Ke, Me, Wie, Wee] = nlebb_elem(Xel,UWel,param,load,flags);

    % Assembly
    if (flags(1)), f(UE(el,:)) = f(UE(el,:)) + fe; end
    if (flags(2)), b(UE(el,:)) = b(UE(el,:)) + be; end
    if (flags(3)), K(UE(el,:),UE(el,:)) = K(UE(el,:),UE(el,:)) + Ke; end
    if (flags(4)), M(UE(el,:),UE(el,:)) = M(UE(el,:),UE(el,:)) + Me; end
    if (flags(5)), Wint = Wint + Wie; end
    if (flags(6)), Wext = Wext + Wee; end
    
end

end


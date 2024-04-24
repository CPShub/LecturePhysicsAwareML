% -------------------------------------------------------------------------
% Physics-aware machine learning
% Cyber-Physical Simulation, TU Darmstadt
% -------------------------------------------------------------------------
% Nonlinear Euler-Bernoulli beam 
% Plot beam deformation (frame for movie)
% -------------------------------------------------------------------------

function fig = nlebb_frame(fig,XX,XE,U,UE,t,npe)

    ne = size(XE,1);
    if (nargin < 7), npe = 3; end

    % Create figure
    if (isempty(fig))
        fig = figure;
        set(fig,'Color','white');
    end
    
    % Evaluate with loop over elements
    pUWx = zeros(npe*ne, 1);
    pUW0 = zeros(npe*ne, 2);
    
    for el = 1:ne
        
        Xel = XX(:,XE(el,:));
        Uel = U(UE(el,:));
        UWe = [Uel(1:2:7)'; Uel(2:2:8)'];

        for ii = 1:npe
            Xi = (ii-1)/(npe-1);
            x = Xel(1) + Xi * (Xel(2)-Xel(1));
            H0 = [1-3*Xi^2+2*Xi^3, Xi-2*Xi^2+Xi^3, 3*Xi^2-2*Xi^3, -Xi^2+Xi^3];  
            pUWx((el-1)*npe+ii, 1) = x;
            pUW0((el-1)*npe+ii, 1) = x + dot(H0, UWe(1,:));
            pUW0((el-1)*npe+ii, 2) = dot(H0, UWe(2,:));
        end
        
    end

    % Plot
    figure(fig); 
    plot(pUW0(:,1),pUW0(:,2),'LineWidth',2);
    title(sprintf('Deformed beam, t=%5.3f s',t));
    grid on;
    box on;
    
end
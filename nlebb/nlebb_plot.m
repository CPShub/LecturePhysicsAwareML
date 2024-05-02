% -------------------------------------------------------------------------
% Physics-aware machine learning
% Cyber-Physical Simulation, TU Darmstadt
% -------------------------------------------------------------------------
% Nonlinear Euler-Bernoulli beam 
% Plot beam deformation and derivatives
% -------------------------------------------------------------------------

function plots = nlebb_plot(plots,XX,XE,U,UE,npe,derivs,it)

    ne = size(XE,1);
    if (nargin < 6), npe = 3; end
    if (nargin < 7), derivs = 0; end
    if (nargin < 8), it = ceil(7*rand()); end

    % Color scheme
    col = 1+rem(it-1, 7);
    cmap = turbo(9);
    cmap = cmap([2 6 3 7 4 8 5],:);

    % Create figure and subplots
    if (isempty(plots))
        plots = figure;
        set(plots,'Color','white');
        subplot(1,derivs+2,1);
        title('Deformed beam');
        hold on;
        grid on;
        subplot(1,derivs+2,2);
        title("u (--), w (-)");
        hold on;
        grid on;
        if (derivs)
            subplot(1,derivs+2,3);
            title("u` (--), w` (-)");
            hold on;
            grid on;
            if (derivs > 1)
                subplot(1,derivs+2,4);
                title("u`` (--), w`` (-)");
                hold on;
                grid on;
            end
        end
    end
    
    % Evaluate with loop over elements
    pUWx = zeros(npe*ne, 1);
    pUW0 = zeros(npe*ne, 3);
    pUW1 = zeros(npe*ne, 2);
    pUW2 = zeros(npe*ne, 2);

    for el = 1:ne
        
        Xel = XX(:,XE(el,:));
        Uel = U(UE(el,:));
        UWe = [Uel(1:2:7)'; Uel(2:2:8)'];

        for ii = 1:npe
            Xi = (ii-1)/(npe-1);
            x = Xel(1) + Xi * (Xel(2)-Xel(1));
            H0 = [1-3*Xi^2+2*Xi^3, Xi-2*Xi^2+Xi^3, 3*Xi^2-2*Xi^3, -Xi^2+Xi^3];  
            pUWx((el-1)*npe+ii, 1) = x;
            pUW0((el-1)*npe+ii, 1) = dot(H0, UWe(1,:));
            pUW0((el-1)*npe+ii, 2) = dot(H0, UWe(2,:));
            pUW0((el-1)*npe+ii, 3) = x + dot(H0, UWe(1,:));
            if (derivs)
                J = abs(Xel(2)-Xel(1));
                H1 = [-6*Xi+6*Xi^2, 1-4*Xi+3*Xi^2, 6*Xi-6*Xi^2, -2*Xi+3*Xi^2] / J;
                pUW1((el-1)*npe+ii, 1) = dot(H1, UWe(1,:));
                pUW1((el-1)*npe+ii, 2) = dot(H1, UWe(2,:));
                if (derivs > 1)
                    H2 = [-6+12*Xi, -4+6*Xi, 6-12*Xi, -2+6*Xi] / J^2;                    
                    pUW2((el-1)*npe+ii, 1) = dot(H2, UWe(1,:));
                    pUW2((el-1)*npe+ii, 2) = dot(H2, UWe(2,:));
                end
            end
        end
        
    end

    % Plot
    plot(plots.Children(derivs+2), pUW0(:,3),pUW0(:,2),...
        'Color',cmap(col,:),'Displayname',sprintf('it=%i',it));
    plot(plots.Children(derivs+1), pUWx(:,1),pUW0(:,1),'--',...
        'Color',cmap(col,:),'Displayname',sprintf('u, it=%i',it));
    plot(plots.Children(derivs+1), pUWx(:,1),pUW0(:,2),...
        'Color',cmap(col,:),'Displayname',sprintf('w, it=%i',it));
    if (derivs)
        plot(plots.Children(derivs), pUWx(:,1),pUW1(:,1),'--',...
            'Color',cmap(col,:),'Displayname',sprintf('u`, it=%i',it));
        plot(plots.Children(derivs), pUWx(:,1),pUW1(:,2),...
            'Color',cmap(col,:),'Displayname',sprintf('w`, it=%i',it));
        if (derivs > 1)
            plot(plots.Children(1),pUWx(:,1),pUW2(:,1),'--', ...
                'Color',cmap(col,:),'Displayname',sprintf('u``, it=%i',it));
            plot(plots.Children(1),pUWx(:,1),pUW2(:,2),...
                'Color',cmap(col,:),'Displayname',sprintf('w``, it=%i',it));
        end
    end

end
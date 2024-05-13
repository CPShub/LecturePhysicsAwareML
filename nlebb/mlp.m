function [YY,SS,DD] = mlp(W,b,activations,x)
L = length(W);
units = zeros(1, L);
units(1) = size(W{1}, 2);
for l=1:L
    units(l+1) = length(b{l});
end

% mlp evaluation
h = cell(1,L);
y = cell(1,L+1);
ydot = cell(1,L);
yddot = cell(1,L);
y{1} = x;
for l=1:L
    h{l} = W{l} * y{l} +  b{l};
    [y{l+1}, ydot{l}, yddot{l}] = activations{l}(h{l});
end
YY = y{end};

if nargout > 1
    % gradient evaluation
    S = cell(1,L+2);
    S{1} = eye(units(1));
    S{end} = eye(units(end));
    SS = S{1};
    for l=1:l
        S{l+1} = ydot{l} .* W{l};
        SS = S{l+1} * SS;
    end
    
    if nargout > 2
        % hessian evaluation
        DD = zeros(units(end),units(1),units(1));
        for l=1:L
            A = S{l+2};
            for i=l+2:L+1
                A = S{i+1} * A;
            end
            B = S{1};
            for i=1:l
                B = S{i} * B;
            end
        
            WSS = W{l} * B;
        
            D = zeros(units(l+1),units(l),units(1));
            for i=1:units(l+1)
                for j=1:units(l)
                    for k=1:units(1)
                        D(i,j,k) = yddot{l}(i) * W{l}(i,j) * WSS(i,k);
                    end
                end
            end
        
            for i=1:units(end)
                for j=1:units(1)
                    for k=1:units(1)
                        DD(i,j,k) = DD(i,j,k) + A(i,:) * D(:,:,k) * B(:,j);
                    end
                end
            end
        end
    end
end
end
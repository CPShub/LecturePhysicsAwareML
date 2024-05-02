function [y, dy] = mlp(w, b, x)
% Initialization of ouputs
y = x;
dy = eye(size(x, 1));

% Loop over all layers
n = length(w);
for i=1:n
    h = w{i} * y + b{i};

    if i < n
        y = softplus(h);
        df = sigmoid(h);
    else
        y = h;
        df = ones(size(h,1), 1);
    end

    dy = (w{i} .* df) * dy;
end

% Activation functions
function y = softplus(x)
    y = log(1 + exp(x));
end
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end
end
function [w, b] = read_weights(units,fname)
    lines = readlines(fname);
    for i = 1:length(lines)-1
        list = split(strip(lines(i), ' '));
        if mod(i, 2) == 0
            b{round(i/2)} = str2double(list);
        else
            w{round((i+1)/2)} = str2double(list)';
        end
    end

    assert(length(w) == length(units)) % assert agreement of layer number
    
    % reshape weights
    for i=1:length(w)
        w{i} = reshape(w{i}, units(i), []);
        % w{i} = reshape(w{i}, [], units(i))';
        % w{i} = reshape(w{i}, units(i), [])';
    end
end


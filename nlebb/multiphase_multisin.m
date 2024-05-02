function y = multiphase_multisin(A,f0,n,i,nph,t)
%MULTISIN Summary of this function goes here
%   Detailed explanation goes here
y = 0;
for k=1:n
    phi_k = -k * (k-1) * pi / n;
    phi_i = 2*pi*(i-1)/nph;
    y = y + A * sin(2*pi*f0*k*t + phi_k + phi_i);
end
end


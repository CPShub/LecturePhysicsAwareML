% -------------------------------------------------------------------------
% Physics-aware machine learning
% Cyber-Physical Simulation, TU Darmstadt
% -------------------------------------------------------------------------
% Plotting of cubic Hermite splines
% -------------------------------------------------------------------------

X = 0:0.02:1;
n = length(X);
Y0 = zeros(n,4);
Y1 = Y0;
Y2 = Y0;

for i=1:n
    Xi = X(i);
    Y0(i,:) = [1-3*Xi^2+2*Xi^3, Xi-2*Xi^2+Xi^3, 3*Xi^2-2*Xi^3, -Xi^2+Xi^3];  
    Y1(i,:) = [-6*Xi+6*Xi^2, 1-4*Xi+3*Xi^2, 6*Xi-6*Xi^2, -2*Xi+3*Xi^2];
    Y2(i,:) = [-6+12*Xi, -4+6*Xi, 6-12*Xi, -2+6*Xi];
end

figure; hold on;
plot(X', Y0(:,1));
plot(X', Y0(:,2));
plot(X', Y0(:,3));
plot(X', Y0(:,4));
% axis equal;
% 
% figure; hold on;
% plot(X', Y1(:,1));
% plot(X', Y1(:,2));
% plot(X', Y1(:,3));
% plot(X', Y1(:,4));
% 
% figure; hold on;
% plot(X', Y2(:,1));
% plot(X', Y2(:,2));
% plot(X', Y2(:,3));
% plot(X', Y2(:,4));

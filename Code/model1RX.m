clear;
clc;
close all;

weeks = 1:49;
cases = [3, 2, 7, 12, 9, 10, 27, 21, 36, 63, 108, 255, 472, 675, 580, 844, 974, 1096, 1354, 1335, 1109, 936, 627, 476, 295, 164, 94, 37, 26, 15, 8, 5, 3, 1, 2, 0, 2, 1, 6, 0, 0, 1, 0, 0, 0, 1, 0, 3, 0];
N = 157759;

I = cases;
T = 49;
S = N*ones(48,1)-I; % N=S+I keeps unchanged

dt=1;
dS=S(2:T)-S(1:T-1);
dI=I(2:T)-I(1:T-1);


b=[dS;dI]/dt;

SI=S(1:T-1).*I(1:T-1);
I=I(1:T-1);

A=[-SI, I;
    SI, -I];

lambda= pinv(A)*b;

beta = lambda(1);
gamma = lambda(2);

% Calculate fitted values
It = zeros(size(weeks));
It(1) = cases(1);
St = N - It(1);
for t = 2:length(weeks)
    dI = (beta*It(t-1)*St - gamma*It(t-1))*dt;
    It(t) = It(t-1) + dI;
    St = N - It(t);
end

% Plot original and fitted values
figure;
plot(weeks, cases, 'bo-', 'DisplayName', 'Original Data');
hold on;
plot(weeks, It, 'ro-', 'DisplayName', 'Fitted Data');
legend('show');
xlabel('Weeks');
ylabel('Number of Cases');
title('Original vs. Fitted Data');

% Calculate RMSE, MeAE, and MaAE
errors = It - cases';
RMSE = sqrt(mean(errors.^2));
MeAE = mean(abs(errors));
MaAE = max(abs(errors));

fprintf('RMSE: %f\n', RMSE);
fprintf('MeAE: %f\n', MeAE);
fprintf('MaAE: %f\n', MaAE);

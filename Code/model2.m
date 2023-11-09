weeks = 0:48;
I = [3, 2, 7, 12, 9, 10, 27, 21, 36, 63, 108, 255, 472, 675, 580, 844, 974, 1096, 1354, 1335, 1109, 936, 627, 476, 295, 164, 94, 37, 26, 15, 8, 5, 3, 1, 2, 0, 2, 1, 6, 0, 0, 1, 0, 0, 0, 1, 0, 3, 0];
N = 157759;

guess = [0.1, 0.1];
options=optimoptions('lsqcurvefit');
options = optimoptions(options,'StepTolerance',1e-10,'FunctionTolerance',1e-10,'MaxFunctionEvaluations',2000);
upper_bound =[inf,inf];
lower_bound =[0,0];
params_fitted = lsqcurvefit(@(params, t) objective(params, t, I), guess, weeks, I, lower_bound, upper_bound, options);
beta_fitted = params_fitted(1)/N;
gamma_fitted = params_fitted(2);

I0 = I(1);
S0 = N - I0;
R0 = 0;
[t, Y] = ode45(@(t,y) SIR_model(t, y, beta_fitted*N, gamma_fitted, N), weeks, [S0; I0; R0]);
figure;
plot(weeks, I, 'r*', t, Y(:,2), 'b-');
xlabel('Weeks');
ylabel('Number of Infected Individuals');
legend('Data', 'Fitted Result');
title('SIR Model');

fprintf('Fitted beta: %.6f\n', beta_fitted);
fprintf('Fitted gamma: %.6f\n', gamma_fitted);
R_0 = beta_fitted*N/gamma_fitted;
fprintf('R_0: %.6f\n', R_0);

rmse = sqrt(mean((I' - Y(:,2)).^2));
meae = mean(abs(I' - Y(:,2)));
maae = max(abs(I' - Y(:,2)));

fprintf('RMSE (Root Mean Squared Error): %.6f\n', rmse);
fprintf('MeAE (Mean Absolute Error): %.6f\n', meae);
fprintf('MaAE (Maximum Absolute Error): %.6f\n', maae);

function dy = SIR_model(t, y, beta, gamma, N)
S = y(1);
I = y(2);
R = y(3);
dy = zeros(3,1);
dy(1) = -beta/N*S*I;
dy(2) = beta/N*S*I - gamma*I;
dy(3) = gamma*I;
end
% Define the objective function for least squares fitting
function I_predicted = objective(params, t, I)
beta = params(1);
gamma = params(2);
N = 157759;
I0 = I(1);
S0 = N - I0;
R0 = 0;
[T, Y] = ode45(@(t,y) SIR_model(t, y, beta, gamma, N), t, [S0; I0; R0]);
I_predicted = interp1(T, Y(:,2), t, 'linear', 'extrap');
end
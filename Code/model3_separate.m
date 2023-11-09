weeks = 0:48;
weeks_train = weeks(1:2:end);
weeks_test = weeks(2:2:end);
I = [3, 2, 7, 12, 9, 10, 27, 21, 36, 63, 108, 255, 472, 675, 580, 844, 974, 1096, 1354, 1335, 1109, 936, 627, 476, 295, 164, 94, 37, 26, 15, 8, 5, 3, 1, 2, 0, 2, 1, 6, 0, 0, 1, 0, 0, 0, 1, 0, 3, 0];
I_train = I(1:2:end);
I_test = I(2:2:end);
N = 157759;

guess =  [0.0001, 1, 0.001, 0.1];
options=optimoptions('lsqcurvefit');
options = optimoptions(options,'StepTolerance',1e-10,'FunctionTolerance',1e-10,'MaxFunctionEvaluations',2000);
upper_bound =[inf,inf, inf, inf];
lower_bound =[0, 0, 0, 0];
params_fitted = lsqcurvefit(@(params, t) objective(params, t, I), guess, weeks_train, I_train, lower_bound, upper_bound, options);
beta_fitted = params_fitted(1);
gamma_fitted = params_fitted(2);
mu_fitted = params_fitted(3);
a_fitted = params_fitted(4);

I0 = I(1);
E0 = 0;
S0 = N - I0;
R0 = 0;
[t, Y] = ode45(@(t,y) SEIR_model(t, y, beta_fitted, gamma_fitted, mu_fitted, a_fitted, N), weeks, [S0; E0; I0; R0]);
figure;
plot(weeks, I, 'r*', t, Y(:,2), 'b-');
xlabel('Weeks');
ylabel('Number of Infected Individuals');
legend('Data', 'Fitted Result');
title('SEIR Model');

fprintf('Fitted beta: %.6f\n', beta_fitted);
fprintf('Fitted gamma: %.6f\n', gamma_fitted);
fprintf('Fitted mu: %.6f\n', mu_fitted);
fprintf('Fitted a: %.6f\n', a_fitted);
R_0 = a_fitted*beta_fitted*N/((mu_fitted + a_fitted)*(mu_fitted + gamma_fitted));
fprintf('R_0: %.6f\n', R_0);

rmse_train = sqrt(mean((I_train' - Y(1:2:end,2)).^2));
meae_train = mean(abs(I_train' - Y(1:2:end,2)));
maae_train = max(abs(I_train' - Y(1:2:end,2)));

rmse_test = sqrt(mean((I_test' - Y(2:2:end,2)).^2));
meae_test = mean(abs(I_test' - Y(2:2:end,2)));
maae_test = max(abs(I_test' - Y(2:2:end,2)));

fprintf('Training RMSE (Root Mean Squared Error): %.6f\n', rmse_train);
fprintf('Training MeAE (Mean Absolute Error): %.6f\n', meae_train);
fprintf('Training MaAE (Maximum Absolute Error): %.6f\n', maae_train);
fprintf('Test RMSE (Root Mean Squared Error): %.6f\n', rmse_test);
fprintf('Test MeAE (Mean Absolute Error): %.6f\n', meae_test);
fprintf('Test MaAE (Maximum Absolute Error): %.6f\n', maae_test);

function dy = SEIR_model(t, y, beta, gamma, mu, a, N)
S = y(1);
E = y(2);
I = y(3);
R = y(4);
dy = zeros(4,1);
dy(1) = mu * N - mu * S - beta * S * I;
dy(2) = beta * S * I - (mu + a) * E;
dy(3) = a * E - (gamma + mu) * I;
dy(4) = gamma * I - mu * R;
end
% Define the objective function for least squares fitting
function I_predicted = objective(params, t, I)
beta = params(1);
gamma = params(2);
mu = params(3);
a = params(4);
N = 157759;
I0 = I(1);
E0 = 0;
S0 = N - I0;
R0 = 0;
[T, Y] = ode45(@(t,y) SEIR_model(t, y, beta, gamma, mu, a, N), t, [S0; E0; I0; R0]);
I_predicted = interp1(T, Y(:,2), t, 'linear', 'extrap');
end
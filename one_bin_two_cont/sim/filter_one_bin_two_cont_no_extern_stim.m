
% This code implements the EM algorithm for the state-space model described
% in the following publication. This version implements the a reduced model
% without an external input.
%
% Wickramasuriya, D. S., & Faghih, R. T. (2019). A Bayesian filtering approach 
% for tracking arousal from binary and continuous skin conductance features. 
% IEEE Transactions on Biomedical Engineering, 67(6), 1749-1760.

close all;
clear;
clc;

%% sample data

load('data_one_bin_two_cont_no_extern_stim.mat');

base_prob = sum(n) / length(n);

%% parameters

M = 1e6;    % maximum iterations
m = 1;
tol = 1e-8; % convergence criteria

b0 = zeros(1, M);   % binary GLM model
b1 = zeros(1, M);

r0 = zeros(1, M);   % continuous model
r1 = zeros(1, M);
vr = zeros(1, M);   % continuous model noise variance (1)

s0 = zeros(1, M);   % continuous model
s1 = zeros(1, M);
vs = zeros(1, M);   % continuous model noise variance (2)

ve = zeros(1, M);   % process noise variance
rho = zeros(1, M);  % random walk forgetting factor

K = length(n);

x_pred = zeros(1, K);
v_pred = zeros(1, K);

x_updt = zeros(1, K);
v_updt = zeros(1, K);

x_smth = zeros(1, K);
v_smth = zeros(1, K);

p_updt = zeros(1, K);

A = zeros(1, K);
W = zeros(1, K);
CW = zeros(1, K);
C = zeros(1, K);

%% initial guesses

b0(1) = log(base_prob / (1 - base_prob));
b1(1) = 1;
r0(1) = r(1); % guess it's the first value of r
r1(1)  = 0.5;
s0(1) = s(1);
s1(1) = 1;
vr(1) = 0.05;
vs(1) = 0.05;
ve(1) = 0.05;
rho(1) = 1;

%% main function

for m = 1:M
    
    for k = 1:K
        
        if (k == 1)
            x_pred(k) = x_smth(1);
            v_pred(k) = ve(m) + ve(m);
        else
            x_pred(k) = rho(m) * x_updt(k - 1);
            v_pred(k) = (rho(m) ^ 2) * v_updt(k - 1) + ve(m);
        end
        
        C(k) = v_pred(k) / (vr(m) * vs(m) + v_pred(k) * ((r1(m) ^ 2) * vs(m) + (s1(m) ^ 2) * vr(m)));
        x_updt(k) = get_posterior_mode(x_pred(k), C(k), r(k), r0(m), r1(m), b0(m), b1(m), vr(m), n(k), s(k), s0(m), s1(m), vs(m));
        
        p_updt(k) = 1 / (1 + exp((-1) * (b0(m) + b1(m) * x_updt(k))));
        v_updt(k) = 1 / ((1 / v_pred(k)) + ((r1(m) ^ 2) / vr(m)) + ((s1(m) ^ 2) / vs(m)) + (b1(m) ^ 2) * p_updt(k) * (1 - p_updt(k)));
    end
    
    x_smth(K) = x_updt(K);
    v_smth(K) = v_updt(K);
    W(K) = v_smth(K) + (x_smth(K) ^ 2);
     
    A(1:(end - 1)) = rho(m) * v_updt(1:(end - 1)) ./ v_pred(2:end); 
    
    for k = (K - 1):(-1):1
       x_smth(k) = x_updt(k) + A(k) * (x_smth(k + 1) - x_pred(k + 1)); 
       v_smth(k) = v_updt(k) + (A(k) ^ 2) * (v_smth(k + 1) - v_pred(k + 1)); 
       
       CW(k) = A(k) * v_smth(k + 1) + x_smth(k) * x_smth(k + 1);
       W(k) = v_smth(k) + (x_smth(k) ^ 2);
    end
    
    prev = [r0(m) r1(m) ve(m) vr(m) rho(m) s0(m) s1(m) vs(m)];

    R = get_linear_parameters(x_smth, W, r, K);
    S = get_linear_parameters(x_smth, W, s, K); 
    
    b0(m + 1) = log(base_prob / (1 - base_prob));
    b1(m + 1) = 1;

    rho(m + 1) = sum(CW) / sum(W(1:end - 1));

    ve(m + 1) = (sum(W(2:end)) + (rho(m +  1) ^ 2) * sum(W(1:(end - 1))) - 2 * rho(m + 1) * sum(CW)) / K;

    r0(m + 1) = R(1, 1);
    r1(m + 1) = R(2, 1);

    s0(m + 1) = S(1, 1);
    s1(m + 1) = S(2, 1);

    vr(m + 1) = get_maximum_variance(r, r0(m + 1), r1(m + 1), W, x_smth, K);
    vs(m + 1) = get_maximum_variance(s, s0(m + 1), s1(m + 1), W, x_smth, K);
    
    next = [r0(m + 1) r1(m + 1) ve(m + 1) vr(m + 1) rho(m + 1) s0(m + 1) s1(m + 1) vs(m +1)];

    mean_dev = mean(abs(next - prev));

    if mean_dev < tol    
        fprintf('Converged at m = %d\n\n', m);
        break;
    else
        fprintf('m = %d\nr0 = %.18f\nr1 = %.18f\nvr = %.18f\n\ns0 = %.18f\ns1 = %.18f\nvs = %.18f\n\nve = %.18f\nrho = %.18f\n\n', ...
        m + 1, r0(m + 1), r1(m + 1), vr(m + 1), s0(m + 1), s1(m + 1), vs(m + 1), ve(m + 1), rho(m + 1));

        x_pred = zeros(1, K);
        v_pred = zeros(1, K);

        x_updt = zeros(1, K);
        v_updt = zeros(1, K);

        x_smth(2:end) = zeros(1, K - 1);    % x_smth(1) needed for next iteration
        v_smth = zeros(1, K);   

        p_updt = zeros(1, K);

        A = zeros(1, K);
        W = zeros(1, K);
        CW = zeros(1, K);
        C = zeros(1, K);
    end 
end

%% calculate confidence limits

p_smth = 1 ./ (1 + exp((-1) * (b0(m) + b1(m) * x_smth)));  % mode, lower and upper confidence limits for binary distribution
r_smth = r0(m) + r1(m) * x_smth;
s_smth = s0(m) + s1(m) * x_smth;

%% plot graphs

figure;

subplot(511);
hold on;
plot(p, 'b');
plot(p_smth, 'r-.', 'linewidth', 1); grid;
plot(find(n == 0) - 1, 1.4 * max(p) * ones(length(find(n == 0))), 'ks', 'MarkerFaceColor', 'k', 'MarkerSize', 4);
plot(find(n == 1) - 1, 1.4 * max(p) * ones(length(find(n == 1))), 'gs', 'MarkerFaceColor', 'g', 'MarkerSize', 4);
ylabel('(a) p_{k}'); ylim([0 0.18]);
title('State Estimation with Simulated Data');

subplot(512);
hold on;
plot(r, 'b');
plot(r_smth, 'r-.', 'linewidth', 1.25); grid;
ylabel('(b) r_{k}'); 

subplot(513);
hold on;
plot(s, 'b');
plot(s_smth, 'r-.', 'linewidth', 1.25); grid;
ylabel('(c) s_{k}');

subplot(514);
hold on;
plot(x, 'b');
plot(x_smth, 'r-.', 'linewidth', 1.25); grid;
ylabel('(d) x_{k}'); xlabel('time index');

subplot(515);
qqplot(x - x_smth);
title('QQ Plot - State Estimate', 'FontWeight', 'Normal');
ylabel('(e) input quantiles');
xlabel('standard normal quantiles');
grid;

%% supplementary functions

function y = get_posterior_mode(x_pred, C, r, r0, r1, b0, b1, vr, n, s, s0, s1, vs)

    M = 200;    % maximum iterations
    
    it = zeros(1, M);
    f = zeros(1, M);
    df = zeros(1, M);
    
    it(1) = x_pred;

    for i = 1:(M - 1)
        f(i) = it(i) - x_pred - C * (r1 * vs * (r - r0 - r1 * x_pred) + s1 * vr * (s - s0 - s1 * x_pred) + ...
            vr * vs * b1 * (n - (1 / (1 + exp((-1) * (b0 + b1 * it(i))))))); 
        df(i) = 1 + C * vr * vs * (b1 ^ 2) * exp(b0 + b1 * it(i)) / ((1 + exp(b0 + b1 * it(i))) ^ 2);
        it(i + 1) = it(i) - f(i) / df(i);
        
        if abs(it(i + 1) - it(i)) < 1e-14
           y = it(i + 1);
           return;
        end
    end
    
    error('Newton-Raphson failed to converge.');

end

function y = get_maximum_variance(z, r0, r1, W, x_smth, K)

    y = (z * z' + K * (r0 ^ 2) + (r1 ^ 2) * sum(W) ...
                - 2 * r0 * sum(z) - 2 * r1 * dot(x_smth, z) + 2 * r0 * r1 * sum(x_smth)) / K;
            
end

function y = get_linear_parameters(x_smth, W, z, K)

    y = [K sum(x_smth); sum(x_smth) sum(W)] \ [sum(z); sum(z .* x_smth)];
    
end




% This code implements the EM algorithm for the state-space model described
% in the following publication. The code implements a version that does not
% estimate the starting state x0 as a separate parameter and in doing so
% permits some bias at the beginning.
% 
% Smith, A. C., Frank, L. M., Wirth, S., Yanike, M., Hu, D., Kubota, Y., ... & Brown, 
% E. N. (2004). Dynamic analysis of learning in behavioral experiments. 
% Journal of Neuroscience, 24(2), 447-461.

close all;
clear;
clc;

load('data_one_bin.mat');

K = length(n);

M = 2e4;
ve = zeros(1, M);   % process noise variance

x_pred = zeros(1, K);
v_pred = zeros(1, K);

x_updt = zeros(1, K);
v_updt = zeros(1, K);

x_smth = zeros(1, K);
v_smth = zeros(1, K);

p_updt = zeros(1, K);

base_prob = sum(n) / length(n);
tol = 1e-6; % convergence criteria

A = zeros(1, K);
W = zeros(1, K);
CW = zeros(1, K);
C = zeros(1, K);

ve(1) = 0.005;
x_smth(1) = 0;
b0 = log(base_prob / (1 - base_prob));

for m = 1:M
    
    for k = 1:K
        
        if (k == 1) % boundary condition
            x_pred(k) = x_smth(1);
            v_pred(k) = ve(m) + ve(m);
        else
            x_pred(k) = x_updt(k - 1);
            v_pred(k) = v_updt(k - 1) + ve(m);
        end

        x_updt(k) = get_state_update(x_pred(k), v_pred(k), b0, n(k));

        p_updt(k) = 1 / (1 + exp((-1) * (b0 + x_updt(k))));
        v_updt(k) = 1 / ((1 / v_pred(k)) + p_updt(k) * (1 - p_updt(k)));
    end
    
    x_smth(K) = x_updt(K);
    v_smth(K) = v_updt(K);
    W(K) = v_smth(K) + (x_smth(K) ^ 2);
     
    A(1:(end - 1)) = v_updt(1:(end - 1)) ./ v_pred(2:end); 
    
    for k = (K - 1):(-1):1
       x_smth(k) = x_updt(k) + A(k) * (x_smth(k + 1) - x_pred(k + 1)); 
       v_smth(k) = v_updt(k) + (A(k) ^ 2) * (v_smth(k + 1) - v_pred(k + 1)); 
       
       CW(k) = A(k) * v_smth(k + 1) + x_smth(k) * x_smth(k + 1);
       W(k) = v_smth(k) + (x_smth(k) ^ 2);
    end
    
    if (m < M)
        
        ve(m + 1) = (sum(W(2:end)) + sum(W(1:(end - 1))) - 2 * sum(CW)) / K;         
        mean_dev = mean(abs(ve(m + 1) - ve(m)));        
            
        if mean_dev < tol 
            fprintf('m = %d\nx0 = %.18f\nve = %.18f\n\n', m, x_smth(1), ve(m));
            fprintf('Converged at m = %d\n\n', m);               
            break;
        else
            fprintf('m = %d\nx0 = %.18f\nve = %.18f\n\n', m, x_smth(1), ve(m + 1));
   
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
end

p_smth = 1 ./ (1 + exp((-1) * (b0 + x_smth)));

figure;
subplot(411);
stem(n, 'fill', 'color', [0 0.75 0]);
ylim([0 1.25]);
ylabel('(a) n_{k}');
grid; title('Estimation with Simulated Data');

subplot(412);
hold on;
plot(p, 'b');
plot(p_smth, 'r-.', 'linewidth', 1.25);
ylabel('(b) p_{k}'); 
grid;

subplot(413);
hold on;
plot(x, 'b');
plot(x_smth, 'r-.', 'linewidth', 1.25);
ylabel('(c) x_{k}'); xlabel('time index'); 
grid;

subplot(414);
qqplot(x - x_smth);
title('QQ Plot - State Estimate', 'FontWeight', 'Normal');
ylabel('(d) input quantiles');
xlabel('standard normal quantiles');
grid;

function [y] = get_state_update(x_pred, v_pred, b0, n)

    M = 50;    % maximum iterations
    
    it = zeros(1, M);
    func = zeros(1, M);
    df = zeros(1, M);
    
    it(1) = x_pred;

    for i = 1:(M - 1)
       func(i) = it(i) - x_pred - v_pred * (n - exp(b0 + it(i)) / (1 + exp(b0 + it(i))));
       df(i) = 1 + v_pred * exp(b0 + it(i)) / ((1 + exp(b0 + it(i))) ^ 2);
       it(i + 1)  = it(i) - func(i) / df(i);
       
       if abs(it(i + 1) - it(i)) < 1e-14 
           y = it(i + 1);
          return
       end
    end
    
    error('Newton-Raphson failed to converge.');

end



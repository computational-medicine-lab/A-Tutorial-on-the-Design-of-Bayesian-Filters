
% This code implements the EM algorithm for the state-space model described
% in the following publication. 
%
% Wickramasuriya, D. S., & Faghih, R. T. (2020). A mixed filter algorithm for 
% sympathetic arousal tracking from skin conductance and heart rate measurements 
% in Pavlovian fear conditioning. Plos one, 15(4), e0231659.
% 
% Note: This code takes considerable time to run. Just for convenience and 
% purposes of illustration, the values in lines 75 to 85 have been adjusted 
% so that they are very close to where the algorithm converges. In general, 
% consider running this code on a cluster. In addition, the code needs to be 
% run for multiple values of eta (line 89) and the best one selected based on the 
% log-likelihood term (line 227).


close all;
clear;
clc;

load('data_one_bin_two_cont_one_spk.mat');

delta = 0.005;

%% parameters

M = 5e5;    % maximum iterations
tol = 1e-5; % convergence criteria

b0 = zeros(1, M);   % binary GLM model
b1 = zeros(1, M);

r0 = zeros(1, M);   % continuous model
r1 = zeros(1, M);
vr = zeros(1, M);   % continuous model noise variance (1)

s0 = zeros(1, M);   % continuous model
s1 = zeros(1, M);
vs = zeros(1, M);   % continuous model noise variance (2)

ve = zeros(1, M);   % process noise variance
rho = zeros(1, M);  % random walk fogetting factor
alpha = zeros(1, M);    % input gain parameter

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

rpeaks = zeros(1, K * 50);
rpeaks(round(rpeak_locs / delta)) = 1;
rpeaks = reshape(rpeaks, [50, K])';

exception_counter = 0;

%% initial guesses

base_prob = sum(n) / length(n);
b0(1) = log(base_prob / (1 - base_prob));
b1(1) = 1;

r0(1) = 0.27154;
r1(1) = 0.5057;
vr(1) = 0.00187;

s0(1) = -0.73899;
s1(1) = 0.25324;
vs(1) = 0.00302;

ve(1) = 0.01883;
rho(1) = 0.99411;
alpha(1) = 0.00818;

theta = theta';

eta = -0.001;

%% main function

for m = 1:M
    
    for k = 1:K
        
        if (k == 1)
            x_pred(k) = x_smth(1);
            v_pred(k) = ve(m) + ve(m);
        else
            x_pred(k) = rho(m) * x_updt(k - 1) + alpha(m) * I(k);
            v_pred(k) = (rho(m) ^ 2) * v_updt(k - 1) + ve(m);
        end
        
        C(k) = v_pred(k) / (vr(m) * vs(m) + v_pred(k) * ((r1(m) ^ 2) * vs(m) + (s1(m) ^ 2) * vr(m)));

        try     % numerical issues can occur due to the integrals
            [temp1, temp2] = get_posterior_mode(x_pred(k), C(k), r(k), r0(m), r1(m), b0(m), b1(m), vr(m), n(k), s(k), s0(m), s1(m), vs(m), ...
                rpeaks(k, :), ul(k, :), delta, w(k, :, :), theta', eta);
            x_updt(k) = temp1;

            p_updt(k) = 1 / (1 + exp((-1) * (b0(m) + b1(m) * x_updt(k))));
            v_updt(k) = 1 / ((1 / v_pred(k)) + ((r1(m) ^ 2) / vr(m)) + ((s1(m) ^ 2) / vs(m)) + (b1(m) ^ 2) * p_updt(k) * (1 - p_updt(k)) - temp2);
        catch
            x_updt(k) = x_pred(k);
            v_updt(k) = v_pred(k);
            exception_counter = exception_counter + 1;
        end
        
        if (mod(k, 100) == 0)
            fprintf('%d ', k);
        end
        
        if (mod(k, 2500) == 0)
            fprintf('\n');
        end
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
    
    if (m < M)
        
        R = get_linear_parameters(x_smth, W, r, K);
        S = get_linear_parameters(x_smth, W, s, K);

        Q = [sum(W(1:end - 1)) (I(2:end) * x_smth(1:(end - 1))'); ...
            (I(2:end) * x_smth(1:(end - 1))') (I * I')] \ [sum(CW); (I(2:end) * x_smth(2:end)')]; 
        
        b0(m + 1) = log(base_prob / (1 - base_prob));
        b1(m + 1) = 1;
        
        rho(m + 1) = Q(1, 1);
        
        if (Q(2, 1) < 0)    % in case this happens (generally only needed with experimental data)
            alpha(m + 1) = alpha(m);
        else
            alpha(m + 1) = Q(2, 1);
        end
        
        ve(m + 1) = (sum(W(2:end)) + (rho(m + 1) ^ 2) * sum(W(1:(end - 1))) - 2 * rho(m + 1) * sum(CW) - ...
            2 * alpha(m + 1) * (I(2:end) * x_smth(2:end)') + 2 * alpha(m + 1) * rho(m + 1) * (I(2:end) * x_smth(1:(end - 1))') + ...
            (alpha(m + 1) ^ 2) * (I * I')) / K;

        r0(m + 1) = R(1, 1);
        r1(m + 1) = R(2, 1);

        s0(m + 1) = S(1, 1);
        s1(m + 1) = S(2, 1);

        vr(m + 1) = get_maximum_variance(r, r0(m + 1), r1(m + 1), W, x_smth, K);
        vs(m + 1) = get_maximum_variance(s, s0(m + 1), s1(m + 1), W, x_smth, K);    

         mean_dev = mean(abs([b0(m + 1) b1(m + 1) r0(m + 1) r1(m + 1) ve(m + 1) vr(m + 1) rho(m + 1) alpha(m + 1) s0(m + 1) s1(m + 1) vs(m + 1)] - ...
            [b0(m) b1(m) r0(m) r1(m) ve(m) vr(m) rho(m) alpha(m) s0(m) s1(m) vs(m)]));       
              
        if mean_dev < tol    
            fprintf('\n\nConverged at m = %d\n\n', m);
            break;
        else
            fprintf('m = %d\nb0 = %.18f\nb1 = %.18f\n\nr0 = %.18f\nr1 = %.18f\nvr = %.18f\n\ns0 = %.18f\ns1 = %.18f\nvs = %.18f\n\nve = %.18f\nrho = %.18f\nalpha = %.18f\n\ndev = %.18f\n\n', ...
            m + 1, b0(m + 1), b1(m + 1), r0(m + 1), r1(m + 1), vr(m + 1), s0(m + 1), s1(m + 1), vs(m + 1), ve(m + 1), rho(m + 1), alpha(m + 1), mean_dev);
        
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

%% calculate confidence limits

p_smth = 1 ./ (1 + exp((-1) * (b0(m) + b1(m) * x_smth))); 
r_smth = r0(m) + r1(m) * x_smth;
s_smth = s0(m) + s1(m) * x_smth;

lambda = zeros(K, 50);
mean_rr = zeros(K, 50);

for i = 1:K
    for j = 1:50
        w1 = [squeeze(w(i, j, :))' [eta x_smth(i)]];
        if (f(theta', ul(i, j), w1) > 1e-18)
            lambda(i, j) = fetch_lambda(theta', ul(i, j), w1);
        end
        mean_rr(i, j) =  mu(theta', w1);
    end
end

lambda_start_index = find(reshape(rpeaks', 1, numel(rpeaks)), 1);
lambda = reshape(lambda', 1, numel(lambda));

ll = get_log_likelihood(eta, rpeaks, ul, delta, w, theta', x_smth, v_smth);
ll_final = sum(nansum(ll));

%% plot graphs

figure;

mean_rr = reshape(mean_rr', 1, numel(mean_rr));
rri = diff(rpeak_locs);
rr_times = rpeak_locs(2:end);

index = (0:(K - 1));
fs_hyp = 4;
t_index = index / fs_hyp;
r_index = ((K - 1):(-1):0); % reverse index
transp = 0.3;

subplot(611);
hold on;
plot(t_index, p, 'b'); grid;
plot(t_index, p_smth, 'r');
plot((find(n == 0) - 1) / fs_hyp, 1.2 * max(p) * ones(length(find(n == 0))), 'ks', 'MarkerFaceColor', 'k', 'MarkerSize', 4);
plot((find(n == 1) - 1) / fs_hyp, 1.2 * max(p) * ones(length(find(n == 1))), 'gs', 'MarkerFaceColor', 'g', 'MarkerSize', 4);
ylabel('(a) p_{k}'); ylim([0 0.25]); 
title('State Estimation with Simulated Data');

subplot(612);
hold on;
plot(t_index, r, 'b'); grid;
plot(t_index, r_smth, 'r');
ylabel('(b) r_{k}');

subplot(613);
hold on;
plot(t_index, s, 'b'); grid;
plot(t_index, s_smth, 'r');
ylabel('(c) s_{k}'); 

subplot(614);
hold on;
plot(t_index, x, 'b'); grid;
plot(t_index, x_smth, 'r');
plot((find(I == 0) - 1) / fs_hyp, (-8) * ones(length(find(I == 0))), 'ks', 'MarkerFaceColor', 'k', 'MarkerSize', 4);
plot((find(I == 1) - 1) / fs_hyp, (-8) * ones(length(find(I == 1))), 'cs', 'MarkerFaceColor', 'c', 'MarkerSize', 4);
ylabel('(d) x_{k}'); 

subplot(615);
hold on;
plot(rr_times, rri, 'o', 'Color', [1, 0.5, 0.25], 'MarkerFaceColor', [1, 0.5, 0.25], 'MarkerSize', 2); grid; 
mu_start_index = round(rpeak_locs(2) / delta);
plot(((0:(length(mean_rr(mu_start_index:end)) - 1)) * delta), mean_rr(mu_start_index:end), 'b');
ylabel('(e) rr_{i}'); xlim([0 t_index(end)]); xlabel('time (s)');

subplot(616);
qqplot(x_smth - x); grid;
title('QQ Plot - State Estimate', 'FontWeight', 'normal');
ylabel('(f) input quantiles');
xlabel('standard normal quantiles');

figure;
get_ks_plot(rpeak_locs, lambda(lambda_start_index:end), delta, 1);
ylabel({'Theoretical', 'Quantiles'}); xlabel('Empirical Quantiles');
title('KS Plot');

%% supplementary functions

function [y, H2] = get_posterior_mode(x_pred, C, r, r0, r1, b0, b1, vr, n, s, s0, s1, vs, rpeaks, ul, delta, w_all, theta, eta)

    M = 200;    % maximum iterations
    
    it = zeros(1, M);
    func = zeros(1, M);
    df = zeros(1, M);
    
    it(1) = x_pred;

    for i = 1:(M - 1)
        
        H1 = zeros(1, 50);
        H2 = zeros(1, 50);
        
        for j = 1:50    % 5 ms -> 0.25 s (4 Hz for skin conductance)
            w = [squeeze(w_all(1, j, :))' [eta it(i)]];
            
            if (f(theta, ul(j), w) > 1e-18) %
                lambda = fetch_lambda(theta, ul(j), w);
                dl_dx = dlambda_dx(theta, ul(j), w);
                
                H1(j) = dl_dx * (rpeaks(j) - lambda * delta) / lambda;
                H2(j) = d2lambda_dx2(theta, ul(j), w) * (rpeaks(j) - lambda * delta) / lambda - rpeaks(j) * (dl_dx ^ 2) / (lambda ^ 2);
            end
        end
        
        H1 = sum(H1);
        H2 = sum(H2);
        
        
        func(i) = it(i) - x_pred - C * (r1 * vs * (r - r0 - r1 * x_pred) + s1 * vr * (s - s0 - s1 * x_pred) + ...
            vr * vs * b1 * (n - (1 / (1 + exp((-1) * (b0 + b1 * it(i)))))) + vr * vs * H1); 
        df(i) = 1 + C * vr * vs * ((b1 ^ 2) * exp(b0 + b1 * it(i)) / ((1 + exp(b0 + b1 * it(i))) ^ 2) - H2);
        it(i + 1) = it(i) - func(i) / df(i);
        
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

function [y] = f(theta, t, w)

    y = sqrt(theta(end) ./ (2 * pi * (t .^ 3))) .* ...
        exp((theta(end) * ((t - mu(theta, w)) .^ 2)) ./ ...
        ((-2) * (mu(theta, w) ^ 2) * t));
    
end

function [y] = intf(theta, t, w)
    
    y = integral(@(t)f(theta, t, w), 0, t);

end

function [y] = mu(theta, w)

    eta = w(end - 1);
    x = w(end);
    p = length(theta) - 2;
    
    y = theta(1) + theta(2:(2 + p - 1)) * w(1:p)' + eta * x;

end

function [y] = fetch_lambda(theta, t, w)
    
    cdf = intf(theta, t, w);
    y = f(theta, t, w) ./ (1 - cdf);

    if (cdf > 1)    % numerical issue
        y = 0;
    end
    
end

function [y] = df_dmu(theta, t, w)

    y = (theta(end) / (mu(theta, w) ^ 3)) * (f(theta, t, w) .* (t - mu(theta, w)));

end

function [y] = df_dx(theta, t, w)

    eta = w(end - 1);
    y = df_dmu(theta, t, w) .* eta;

end

function [y] = intdf_dx(theta, t, w)

    y = integral(@(t)df_dx(theta, t, w), 0, t);

end

function [y] = dlambda_dx(theta, t, w)

    cdf = intf(theta, t, w);
    
    if (cdf > 1)    % numerical issue
        y = 0;
    else        
        y = ((1 - cdf) .* df_dx(theta, t, w) + ...
            f(theta, t, w) .* intdf_dx(theta, t, w)) ./ ((1 - cdf) .^ 2);
    end
end

function [y] = d2f_dmu2(theta, t, w)

    y = theta(end) * (df_dmu(theta, t, w) .* ((t - mu(theta, w)) / (mu(theta, w) ^ 3)) + ...
        f(theta, t, w) .* ((2 * mu(theta, w) - 3 * t) / (mu(theta, w) ^ 4)));

end

function [y] = d2f_dx2(theta, t, w)

    eta = w(end - 1);
    y = d2f_dmu2(theta, t, w) .* (eta ^ 2);

end

function [y] = intd2f_dx2(theta, t, w)

    y = integral(@(t)d2f_dx2(theta, t, w), 0, t);

end

function [y] = d2lambda_dx2(theta, t, w)

    y = (2 * dlambda_dx(theta, t, w) * (1 - intf(theta, t, w)) * intdf_dx(theta, t, w) + ...
         d2f_dx2(theta, t, w) * (1 - intf(theta, t, w)) + ...
         f(theta, t, w) * intd2f_dx2(theta, t, w)) / ((1 - intf(theta, t, w)) ^ 2);

end

function [y] = get_log_likelihood(eta, rpeaks, ul, delta, w_all, theta, x, v)

    K = length(x);
    y = zeros(K, 50);
    
    for k = 1:K
        for j = 1:50
            w = [squeeze(w_all(k, j, :))' [eta x(k)]];
            
            if (f(theta, ul(k, j), w) > 1e-18) 
                
                lambda = fetch_lambda(theta, ul(k, j), w);  
                dl_dx = dlambda_dx(theta, ul(k, j), w);
                d2l_dx2 = d2lambda_dx2(theta, ul(k, j), w);
                nkj = rpeaks(k, j);
                
                y(k, j) = nkj * log(delta * lambda) - delta * lambda + ...
                    (d2l_dx2 * (nkj - lambda * delta) / lambda - nkj * (dl_dx ^ 2) / (lambda ^ 2)) * v(k) * 0.5;
            end
            
        end
    end
    
end



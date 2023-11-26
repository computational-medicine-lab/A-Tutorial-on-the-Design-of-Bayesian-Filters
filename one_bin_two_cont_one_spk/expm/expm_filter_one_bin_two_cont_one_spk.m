
% This code implements the EM algorithm for the state-space model described
% in the following publication. 
%
% Wickramasuriya, D. S., & Faghih, R. T. (2020). A mixed filter algorithm for 
% sympathetic arousal tracking from skin conductance and heart rate measurements 
% in Pavlovian fear conditioning. Plos one, 15(4), e0231659.
% 
% Note: This code takes considerable time to run. The overfitting control check
% and the form of early stopping help somewhat (line 184). In general, 
% consider running this code on a cluster. In addition, the code needs to be 
% run for multiple values of eta (line 106) and the best one selected based on the 
% log-likelihood term (line 261).

close all;
clear;
clc;

load('expm_data_one_bin_two_cont_one_spk.mat');

delta = 0.005;
min_scr_thresh = 0.015;
min_scr_prom = min_scr_thresh;
fs = 4;
epoch = 10;

stim = s_data.aug_stim;
ph = s_data.ph;
tn = s_data.tn;
rpeaks = s_data.rpeaks;
ul = s_data.ul;

[pks, locs] = findpeaks(ph, 'MinPeakHeight', min_scr_thresh, 'MinPeakProminence', min_scr_prom);
r = interp1([1 locs length(ph)], log([ph(find(ph > 0, 1)) pks ph(end)]), 1:length(s_data.ph), 'cubic');

s = tn;
n = zeros(1, length(r));
I = zeros(1, length(r));

n(locs) = 1;
I(stim) = 1;

std_s = std(s);
std_r = std(r);

s = s / std_s;
r = r / std_r;

%% parameters

M = 5e5;    % maximum iterations
tol = 1e-6; % convergence criteria

b0 = zeros(1, M);   % binary GLM model
b1 = zeros(1, M);

r0 = zeros(1, M);   % continuous GLM model
r1 = zeros(1, M);
vr = zeros(1, M);   % continuous GLM model noise variance (1)

s0 = zeros(1, M);   % continuous GLM
s1 = zeros(1, M);
vs = zeros(1, M);   % continuous GLM model noise variance (2)

ve = zeros(1, M);   % process noise variance
rho = zeros(1, M);  % random walk correlation
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

exception_counter = 0;

%% initial guesses

base_prob = sum(n) / length(n);
b0(1) = log(base_prob / (1 - base_prob));
b1(1) = 1;
r0(1) = r(1); % guess it's the first value of r
r1(1)  = 1;
s0(1) = s(1);
s1(1) = 1;
vr(1) = 0.05;
vs(1) = 0.05;
ve(1) = 0.05;
rho(1) = 1;
alpha(1) = 0.5;

theta = s_data.theta;

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
                rpeaks(k, :), ul(k, :), delta, s_data.w(k, :, :), theta', eta);
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
        
        bb = fsolve(@(b) binary_parameter_derivatives(b, n, x_smth, v_smth), [-5 1], optimset('Display','off'));
        b0(m + 1) = bb(1);
        b1(m + 1) = bb(2);
        
        rho(m + 1) = Q(1, 1);
        
        if (Q(2, 1) < 0)    % check in case this happens
            alpha(m + 1) = alpha(m);
        else
            alpha(m + 1) = Q(2, 1);
        end
        
        ve(m + 1) = (sum(W(2:end)) + (rho(m + 1) ^ 2) * sum(W(1:(end - 1))) - 2 * rho(m + 1) * sum(CW) - ...
            2 * alpha(m + 1) * (I(2:end) * x_smth(2:end)') + 2 * alpha(m + 1) * rho(m + 1) * (I(2:end) * x_smth(1:(end - 1))') + ...
            (alpha(m + 1) ^ 2) * (I * I')) / K;
        
        if (abs(get_maximum_variance(r, R(1, 1), R(2, 1), W, x_smth, K) - get_maximum_variance(s, S(1, 1), S(2, 1), W, x_smth, K)) > 0.1)   % terminate once overfitting is detected 
            break;
        else          
            r0(m + 1) = R(1, 1);
            r1(m + 1) = R(2, 1);

            s0(m + 1) = S(1, 1);
            s1(m + 1) = S(2, 1);

            vr(m + 1) = get_maximum_variance(r, r0(m + 1), r1(m + 1), W, x_smth, K);
            vs(m + 1) = get_maximum_variance(s, s0(m + 1), s1(m + 1), W, x_smth, K);    
           
        end

         mean_dev = mean(abs([b0(m + 1) b1(m + 1) r0(m + 1) r1(m + 1) ve(m + 1) vr(m + 1) rho(m + 1) alpha(m + 1) s0(m + 1) s1(m + 1) vs(m + 1)] - ...
            [b0(m) b1(m) r0(m) r1(m) ve(m) vr(m) rho(m) alpha(m) s0(m) s1(m) vs(m)]));       
              
        if mean_dev < tol    
            fprintf('Converged at m = %d\n\n', m);
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
lcl_fp = zeros(1, K); 
ucl_fp = zeros(1, K);

r_smth = exp((r0(m) + r1(m) * x_smth) * std_r);
s_smth = (s0(m) + s1(m) * x_smth) * std_s;

skn_avg = get_trial_averages(s_data, x_smth, epoch, fs, 'skn');
x_avg = get_trial_averages(s_data, x_smth, epoch, fs, 'x_smth');

t_epoch = ((-1):(1 / fs):(epoch - 1 - (1 / fs)));
tr_epoch = ((epoch - 1 - (1 / fs)):(-1 / fs):(-1));

fprintf('Plotting\n');

lambda = zeros(K, 50);
mean_rr = zeros(K, 50);

for i = 1:K
    for j = 1:50
        w = [squeeze(s_data.w(i, j, :))' [eta x_smth(i)]];
        if (f(theta', ul(i, j), w) > 1e-18)
            lambda(i, j) = fetch_lambda(theta', ul(i, j), w);
        end
        mean_rr(i, j) =  mu(theta', w);
    end
end

lambda_start_index = find(reshape(s_data.rpeaks', 1, numel(s_data.rpeaks)), 1);
lambda = reshape(lambda', 1, numel(lambda));

ll = get_log_likelihood(eta, rpeaks, ul, delta, s_data.w, theta', x_smth, v_smth);
ll_final = sum(nansum(ll));

%% plot graphs

mean_rr = reshape(mean_rr', 1, numel(mean_rr));
rri = diff(s_data.rpeak_locs);
rr_times = s_data.rpeak_locs(2:end);

index = (0:(K - 1));
t_index = index / fs;
r_index = ((K - 1):(-1):0);
transp = 0.3;

subplot(711);
plot(t_index, s_data.x, 'color', [(102 / 255) 0 (204 / 255)]);
ylabel('(a) z_{k}'); grid; xlim([0 t_index(end)]);
set(gca,'xticklabel', []);
ylim([(min(s_data.x) - 0.1) (max(s_data.x) + 0.1)]);
title('State Estimation with Experimental Data');

subplot(712);
hold on;
plot(find(n == 0) / fs, max(p_smth) * 1.3 * ones(length(find(n == 0))), 'ks', 'MarkerFaceColor', 'k', 'MarkerSize', 3);
plot(find(n == 1) / fs, max(p_smth) * 1.3 * ones(length(find(n == 1))), 'gs', 'MarkerFaceColor', 'g', 'MarkerSize', 3);
plot(t_index, p_smth, 'r');
   
ylabel('(b) p_{k}'); 
xlim([0 t_index(end)]); ylim([0 (max(p_smth) * 1.5)]); grid;
set(gca,'xticklabel', []);

subplot(713);
hold on;
plot(t_index, r_smth, ':', 'color', [0 0.3 0], 'linewidth', 1.5);
plot(t_index, exp(r * std_r), 'color', [0 0.9 0]); 
   
ylabel('(c) e^{r_{k}}'); grid;
xlim([0 t_index(end)]); 
set(gca,'xticklabel', []);

subplot(714);
hold on;
plot(t_index, s_smth, ':', 'color', [0.5 (25 / 255) (66 / 255)], 'linewidth', 1.5);
plot(t_index, s * std_s, 'color', [1 0.5 (179 / 255)]);
   
ylabel('(d) s_{k}');
xlim([0 t_index(end)]); grid;
set(gca,'xticklabel', []);

subplot(715);
hold on;
plot(t_index, x_smth, 'color', 'b');
plot(find(I == 0) / fs, (min(x_smth) - 0.5) * ones(length(find(I == 0))), 'ks', 'MarkerFaceColor', 'k', 'MarkerSize', 3);
plot(find(I == 1) / fs, (min(x_smth) - 0.5) * ones(length(find(I == 1))), 'cs', 'MarkerFaceColor', 'c', 'MarkerSize', 3);
   
ylabel('(e) x_{k}'); ylim([(min(x_smth) - 1) (max(x_smth) + 1)]);
xlim([0 t_index(end)]); grid; set(gca,'xticklabel', []);

subplot(716);
hold on;
plot(rr_times / 60, rri, 'o', 'Color', [1, 0.5, 0.25], 'MarkerFaceColor', [1, 0.5, 0.25], 'MarkerSize', 3); grid; 
mu_start_index = round(s_data.rpeak_locs(2) / delta);
plot(((0:(length(mean_rr(mu_start_index:end)) - 1)) * delta) / 60, mean_rr(mu_start_index:end), 'b');
ylabel('(f) rr_{i}'); xlim([0 t_index(end)] / 60); xlabel('time (min)');

subplot(7, 2, 13);
hold on;

plot(t_epoch, skn_avg(7, :), 'color', [0 0.8 0], 'linewidth', 1.5);
fill([t_epoch, tr_epoch], [skn_avg(8, :) fliplr(skn_avg(9, :))], 'g', 'EdgeColor', 'none', 'FaceAlpha', 0.2);

plot(t_epoch, skn_avg(4, :), 'm', 'linewidth', 1.5);
fill([t_epoch, tr_epoch], [skn_avg(5, :) fliplr(skn_avg(6, :))], 'm', 'EdgeColor', 'none', 'FaceAlpha', 0.2); 

plot(t_epoch, skn_avg(1, :), 'r', 'linewidth', 1.5);
fill([t_epoch, tr_epoch], [skn_avg(2, :) fliplr(skn_avg(3, :))], 'r', 'EdgeColor', 'none', 'FaceAlpha', 0.2);      

xlim([t_epoch(1) t_epoch(end)]);

grid;
xlabel('time (s)'); ylabel('(g) z_{k}');

subplot(7, 2, 14);
hold on;

plot(t_epoch, x_avg(7, :), 'color', [0 0.8 0], 'linewidth', 1.5);
fill([t_epoch, tr_epoch], [x_avg(8, :) fliplr(x_avg(9, :))], 'g', 'EdgeColor', 'none', 'FaceAlpha', 0.2);

plot(t_epoch, x_avg(4, :), 'm', 'linewidth', 1.5);
fill([t_epoch, tr_epoch], [x_avg(5, :) fliplr(x_avg(6, :))], 'm', 'EdgeColor', 'none', 'FaceAlpha', 0.2); 

plot(t_epoch, x_avg(1, :), 'r', 'linewidth', 1.5);
fill([t_epoch, tr_epoch], [x_avg(2, :) fliplr(x_avg(3, :))], 'r', 'EdgeColor', 'none', 'FaceAlpha', 0.2);      

xlim([t_epoch(1) t_epoch(end)]);

grid;
xlabel('time (s)'); ylabel('(h) x_{k}');

figure;
get_ks_plot(s_data.rpeak_locs, lambda(lambda_start_index:end), delta, 1);

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

function y = binary_parameter_derivatives(b, n, x_smth, v_smth)
    
    y = zeros(1, 2);
    K = length(n);
    
    b0 = b(1);
    b1 = b(2);
    p = zeros(1, K);
    
    for k = 1:K
        p(k) = 1 / (1 + exp((-1) * (b0 + b1 * x_smth(k))));
        y(1) = y(1) + n(k) - p(k) - 0.5 * v_smth(k) * (b1 ^ 2) * p(k) * (1 - p(k)) * (1 - 2 * p(k));
        y(2) = y(2) + n(k) * x_smth(k) - x_smth(k) * p(k) - 0.5 * v_smth(k) * b1 * p(k) * (1 - p(k)) * (2 + x_smth(k) * b1 * (1 - 2 * p(k)));
    end
    
end

function y = get_maximum_variance(z, r0, r1, W, x_smth, K)

    y = (z * z' + K * (r0 ^ 2) + (r1 ^ 2) * sum(W) ...
                - 2 * r0 * sum(z) - 2 * r1 * dot(x_smth, z) + 2 * r0 * r1 * sum(x_smth)) / K;
            
end

function y = get_linear_parameters(x_smth, W, z, K)

    y = [K sum(x_smth); sum(x_smth) sum(W)] \ [sum(z); sum(z .* x_smth)];
    
end

function y = get_trial_averages(s, x_smth, epoch, fs, option)

    y = zeros(9, epoch * fs); 

    csm_ep = zeros(length(s.csm), epoch * fs);
    csp_us_ep = zeros(length(s.csp_us), epoch * fs);
    csp_nus_ep = zeros(length(s.csp_nus), epoch * fs);

    csm = s.csm;
    csp_us = s.csp_us;
    csp_nus = s.csp_nus;

    if strcmp(option, 'x_smth')
        
        for j = 1:length(csm)
            csm_ep(j, :) = x_smth((s.stim(csm(j)) - fs ):(s.stim(csm(j)) + 9 * fs - 1));
        end

        for j = 1:length(csp_us)
            csp_us_ep(j, :) = x_smth((s.stim(csp_us(j)) - fs):(s.stim(csp_us(j)) + 9 * fs - 1));
        end

        for j = 1:length(csp_nus)
            csp_nus_ep(j, :) = x_smth((s.stim(csp_nus(j)) - fs):(s.stim(csp_nus(j)) + 9 * fs - 1));
        end
        
    elseif strcmp(option, 'skn') 

        for j = 1:length(csm)
            csm_ep(j, :) = s.x((s.stim(csm(j)) - fs ):(s.stim(csm(j)) + 9 * fs - 1));
        end

        for j = 1:length(csp_us)
            csp_us_ep(j, :) = s.x((s.stim(csp_us(j)) - fs):(s.stim(csp_us(j)) + 9 * fs - 1));
        end

        for j = 1:length(csp_nus)
            csp_nus_ep(j, :) = s.x((s.stim(csp_nus(j)) - fs):(s.stim(csp_nus(j)) + 9 * fs - 1));
        end        
        
    end
    
    y(1, :) = mean(csp_us_ep);
    y(2, :) = mean(csp_us_ep) + tinv(0.975, length(csp_us) - 1) * std(csp_us_ep) / sqrt(length(csp_us));
    y(3, :) = mean(csp_us_ep) + tinv(0.025, length(csp_us) - 1) * std(csp_us_ep) / sqrt(length(csp_us));
    
    y(4, :) = mean(csp_nus_ep);
    y(5, :) = mean(csp_nus_ep) + tinv(0.975, length(csp_nus) - 1) * std(csp_nus_ep) / sqrt(length(csp_nus));
    y(6, :) = mean(csp_nus_ep) + tinv(0.025, length(csp_nus) - 1) * std(csp_nus_ep) / sqrt(length(csp_nus));    
    
    y(7, :) = mean(csm_ep);
    y(8, :) = mean(csm_ep) + tinv(0.975, length(csm) - 1) * std(csm_ep) / sqrt(length(csm));
    y(9, :) = mean(csm_ep) + tinv(0.025, length(csm) - 1) * std(csm_ep) / sqrt(length(csm)); 
    
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



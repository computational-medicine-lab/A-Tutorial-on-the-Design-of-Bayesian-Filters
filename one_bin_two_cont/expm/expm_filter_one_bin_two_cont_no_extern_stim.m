
% This code implements the EM algorithm for the state-space model described
% in the following publication. This version implements the a reduced model
% without an external input. Note that overfitting control and a form of 
% early stopping are applied (line 142). 
%
% Wickramasuriya, D. S., & Faghih, R. T. (2019). A Bayesian filtering approach 
% for tracking arousal from binary and continuous skin conductance features. 
% IEEE Transactions on Biomedical Engineering, 67(6), 1749-1760.

close all;
clear;
clc;

%% sample data

load('expm_data_one_bin_two_cont_no_extern_stim.mat');

min_scr_thresh = 0.015;
min_scr_prom = min_scr_thresh;
fs = 2;

t = (0:(length(phasic) - 1)) / fs;

ph = phasic;
tn = tonic;
x_orig = y;

[pks, locs] = findpeaks(ph, 'MinPeakHeight', min_scr_thresh, 'MinPeakProminence', min_scr_prom);

r = interp1([1 locs length(ph)], log([ph(1) pks ph(end)]), 1:length(ph), 'cubic');
s = tn;
n = zeros(1, length(r));
I = zeros(1, length(r));
n(locs) = 1;

base_prob = sum(n) / length(n);

std_s = std(s);
std_r = std(r);

s = s / std_s;
r = r / std_r;

%% parameters

M = 5e5;    % maximum iterations
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
r1(1)  = 1;
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
    
    if (m < M)
        
        R = get_linear_parameters(x_smth, W, r, K);
        S = get_linear_parameters(x_smth, W, s, K); 
        
        b0(m + 1) = log(base_prob / (1 - base_prob));
        b1(m + 1) = 1;
        
        rho(m + 1) = sum(CW) / sum(W(1:end - 1)); 
        
        ve(m + 1) = (sum(W(2:end)) + (rho(m + 1) ^ 2) * sum(W(1:(end - 1))) - 2 * rho(m + 1) * sum(CW)) / K;
        
        if (abs(get_maximum_variance(r, R(1, 1), R(2, 1), W, x_smth, K) - get_maximum_variance(s, S(1, 1), S(2, 1), W, x_smth, K)) > 0.1)   % overfitting check
            r0(m + 1) = r0(m);
            r1(m + 1) = r1(m);

            s0(m + 1) = s0(m);
            s1(m + 1) = s1(m);

            vr(m + 1) = vr(m);
            vs(m + 1) = vs(m);   
            
            mean_dev = mean(abs([ve(m + 1) rho(m + 1)] - [ve(m) rho(m)]));
        else          
            r0(m + 1) = R(1, 1);
            r1(m + 1) = R(2, 1);

            s0(m + 1) = S(1, 1);
            s1(m + 1) = S(2, 1);

            vr(m + 1) = get_maximum_variance(r, r0(m + 1), r1(m + 1), W, x_smth, K);
            vs(m + 1) = get_maximum_variance(s, s0(m + 1), s1(m + 1), W, x_smth, K);    
            
            mean_dev = mean(abs([r0(m + 1) r1(m + 1) ve(m + 1) vr(m + 1) rho(m + 1) s0(m + 1) s1(m + 1) vs(m + 1)] - ...
                [r0(m) r1(m) ve(m) vr(m) rho(m) s0(m) s1(m) vs(m)]));
        end
            
        if mean_dev < tol    
            fprintf('Converged at m = %d\n\n', m);
            break;
        else
            fprintf('m = %d\nb0 = %.18f\nb1 = %.18f\n\nr0 = %.18f\nr1 = %.18f\nvr = %.18f\n\ns0 = %.18f\ns1 = %.18f\nvs = %.18f\n\nve = %.18f\nrho = %.18f\n\n', ...
            m + 1, b0(m + 1), b1(m + 1), r0(m + 1), r1(m + 1), vr(m + 1), s0(m + 1), s1(m + 1), vs(m + 1), ve(m + 1), rho(m + 1));
        
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

p_smth = 1 ./ (1 + exp((-1) * (b0(m) + b1(m) * x_smth)));  % mode, lower and upper confidence limits for binary distribution
r_smth = exp(std_r * (r0(m) + r1(m) * x_smth));

s_smth = (s0(m) + s1(m) * x_smth) * std_s;

lcl_x = norminv(0.025, x_smth, sqrt(v_smth));
ucl_x = norminv(0.975, x_smth, sqrt(v_smth));

lcl_p = zeros(1, K);
ucl_p = zeros(1, K);

for k = 1:K
    [lcl_p(k), ucl_p(k)] = get_pk_conf_lims(v_smth(k), b0(m), x_smth(k));
end

certainty = 1 - normcdf(prctile(x_smth, 50) * ones(1, length(x_smth)), x_smth, sqrt(v_smth));

%% plot graphs
disp('Plotting...');

xp_fs_plot = 4;

index = (0:(K - 1));
t_index = index / fs;
r_index = ((K - 1):(-1):0) / fs; 
transp = 0.3;

subplot(611);
hold on;
plot(t_index, x_orig, 'k', 'linewidth', 1.25);
plot(find(n == 0) / fs, 3.7 * ones(length(find(n == 0))), 'ks', 'MarkerFaceColor', 'k', 'MarkerSize', 5); 
plot(find(n == 1) / fs, 3.7 * ones(length(find(n == 1))), 'gs', 'MarkerFaceColor', 'g', 'MarkerSize', 5);
ylim([0 4]); yl = ylim;

patch([xp(1), xp(2), xp(2), xp(1)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(2), xp(3), xp(3), xp(2)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], 'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(3), xp(4), xp(4), xp(3)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], [1 0.647059 0], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(4), xp(5), xp(5), xp(4)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(5), xp(6), xp(6), xp(5)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], 'y', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

ylabel({'(a) skin cond.', '(\mu S)'}); grid; xlim([0 (xp(6) / xp_fs_plot)]);
set(gca,'xticklabel', []);
title('State Estimation with Experimental Data');

subplot(612);
hold on;

plot(t_index, r_smth, ':', 'color', [0 0.3 0], 'linewidth', 1.5);
plot(t_index, exp(r * std_r), 'color', [0 0.9 0], 'linewidth', 1.5); 
grid;
   
xlim([0 (xp(6) / xp_fs_plot)]); 
ylim([(min([exp(r * std_r) r_smth]) - 0.25) (0.25 +  max([exp(r * std_r) r_smth]))]);  yl = ylim;

patch([xp(1), xp(2), xp(2), xp(1)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(2), xp(3), xp(3), xp(2)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], 'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(3), xp(4), xp(4), xp(3)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], [1 0.647059 0], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(4), xp(5), xp(5), xp(4)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(5), xp(6), xp(6), xp(5)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], 'y', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

set(gca,'xticklabel', []);
ylabel('(b) phasic');

subplot(613);
hold on;

plot(t_index, s_smth, ':', 'color', [0.5 (25 / 255) (66 / 255)], 'linewidth', 1.5);
plot(t_index, s * std_s, 'color', [1 0.5 (179 / 255)], 'linewidth', 1.5); grid;
   
xlim([0 (xp(6) / xp_fs_plot)]); 
ylim([(min([(s * std_s) s_smth]) - 0.25) (0.25 + max([(s * std_s) s_smth]))]); yl = ylim;

patch([xp(1), xp(2), xp(2), xp(1)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(2), xp(3), xp(3), xp(2)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], 'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(3), xp(4), xp(4), xp(3)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], [1 0.647059 0], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(4), xp(5), xp(5), xp(4)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(5), xp(6), xp(6), xp(5)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], 'y', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

set(gca,'xticklabel', []);
ylabel('(c) tonic');

subplot(614);
hold on;
plot(t_index, x_smth, 'color', 'b', 'linewidth', 1.25); grid;
fill([t_index, r_index], [lcl_x fliplr(ucl_x)], 'c', 'EdgeColor', 'none', 'FaceAlpha', 0.5);
ylim([(min(x_smth) - 0.25) (0.25 + max(x_smth))]); yl = ylim;

patch([xp(1), xp(2), xp(2), xp(1)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(2), xp(3), xp(3), xp(2)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], 'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(3), xp(4), xp(4), xp(3)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], [1 0.647059 0], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(4), xp(5), xp(5), xp(4)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(5), xp(6), xp(6), xp(5)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], 'y', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

xlim([0 (xp(6) / xp_fs_plot)]);

set(gca,'xticklabel', []);
ylabel('(d) state (x_{k})');

subplot(615);
hold on;   
plot(t_index, p_smth, 'r', 'linewidth', 1.5); grid;
fill([t_index, r_index], [lcl_p fliplr(ucl_p)], [1, 0, (127 / 255)], 'EdgeColor', 'none', 'FaceAlpha', 0.3);   

xlim([0 (xp(6) / xp_fs_plot)]); 
ylim([0 (max(p_smth) * 1.5)]); yl = ylim;

patch([xp(1), xp(2), xp(2), xp(1)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(2), xp(3), xp(3), xp(2)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], 'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(3), xp(4), xp(4), xp(3)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], [1 0.647059 0], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(4), xp(5), xp(5), xp(4)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(5), xp(6), xp(6), xp(5)] / xp_fs_plot, [yl(1) yl(1) yl(2) yl(2)], 'y', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

set(gca,'xticklabel',[]);
ylabel({'(e) probability', '(p_{k})'}, 'FontSize', 11);

subplot(616);
hold on;
v1 = [0 0.9; t(end) 0.9; t(end) 1; 0 1];
c1 = [1 (220 / 255) (220 / 255); 1 (220 / 255) (220 / 255); 1 0 0; 1 0 0];
faces1 = [1 2 3 4];

patch('Faces', faces1, 'Vertices', v1, 'FaceVertexCData', c1, 'FaceColor', 'interp', ...
    'EdgeColor', 'none', 'FaceAlpha', 0.7);

v2 = [0 0; t(end) 0; t(end) 0.1; 0 0.1];
c2 = [0 0.8 0; 0 0.8 0; (204 / 255) 1 (204 / 255); (204 / 255) 1 (204 / 255)];
faces2 = [1 2 3 4];

patch('Faces', faces2, 'Vertices', v2, 'FaceVertexCData', c2, 'FaceColor', 'interp', ...
    'EdgeColor', 'none', 'FaceAlpha', 0.7);

plot(t, certainty, 'color', [(138 / 255) (43 / 255) (226 / 255)], 'linewidth', 1.5); grid; 
xlim([0 (xp(6) / xp_fs_plot)]);

xlabel('time (s)'); 
ylabel('(f) HAI');

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

function [lcl, ucl] = get_pk_conf_lims(v, b0, x)

    p = (1e-4:1e-4:1);
  
    fp = cumtrapz(p, 1 ./ (sqrt(2 * pi * v) * p .* (1 - p)) .* ...
        exp(((-1) / (2 * v))* (log(p ./ ((1 - p) * exp(b0))) - x) .^ 2));
    
    n = find(fp <= 0.975);
    m = find(fp < 0.025);
    
    ucl = p(n(end));
    lcl = p(m(end));
end


function y = get_maximum_variance(z, r0, r1, W, x_smth, K)

    y = (z * z' + K * (r0 ^ 2) + (r1 ^ 2) * sum(W) ...
                - 2 * r0 * sum(z) - 2 * r1 * dot(x_smth, z) + 2 * r0 * r1 * sum(x_smth)) / K;
            
end

function y = get_linear_parameters(x_smth, W, z, K)

    y = [K sum(x_smth); sum(x_smth) sum(W)] \ [sum(z); sum(z .* x_smth)];
    
end


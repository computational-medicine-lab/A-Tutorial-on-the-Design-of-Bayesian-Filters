

% This code implements the EM algorithm for the state-space model described
% in the following publication. This is a slightly different version of the 
% code than that provided for simulated data. This one does not permit as 
% much bias at the beginning since it estimates the initial state x0 as 
% a separate model parameter. 
% 
% Smith, A. C., Frank, L. M., Wirth, S., Yanike, M., Hu, D., Kubota, Y., ... & Brown, 
% E. N. (2004). Dynamic analysis of learning in behavioral experiments. 
% Journal of Neuroscience, 24(2), 447-461.
%
% This version of the code is closer to what was used to estimate sympathetic 
% arousal from skin conductance as described in the following publications. 
%
% Wickramasuriya, D. S., Amin, M., & Faghih, R. T. (2019). Skin conductance 
% as a viable alternative for closing the deep brain stimulation loop in 
% neuropsychiatric disorders. Frontiers in neuroscience, 13, 780.
%
% Wickramasuriya, D. S., Qi, C., & Faghih, R. T. (2018, July). A state-space 
% approach for detecting stress from electrodermal activity. In 2018 40th Annual 
% International Conference of the IEEE Engineering in Medicine and Biology 
% Society (EMBC) (pp. 3562-3567). IEEE.

close all;
clear;
clc;

load('expm_data_one_bin.mat');

K = length(u);
n = zeros(1, K);

pt = find(u > 0);
n(pt) = 1;

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
tol = 1e-8; % convergence criteria

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
    x0_prev = x_smth(1);
    
    for k = (K - 1):(-1):1
       x_smth(k) = x_updt(k) + A(k) * (x_smth(k + 1) - x_pred(k + 1)); 
       v_smth(k) = v_updt(k) + (A(k) ^ 2) * (v_smth(k + 1) - v_pred(k + 1)); 
       
       CW(k) = A(k) * v_smth(k + 1) + x_smth(k) * x_smth(k + 1);
       W(k) = v_smth(k) + (x_smth(k) ^ 2);
    end
    
    if (m < M)
        
        ve(m + 1) = (sum(W(2:end)) + sum(W(1:(end - 1))) - 2 * sum(CW) + 0.5 * W(1)) / (K + 1); 
        x0 = x_smth(1) / 2;      
            
        if (abs(ve(m + 1) - ve(m)) < tol) && (abs(x0 - x0_prev) < tol)
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
            x_smth(1) = x0;
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

lcl_x = norminv(0.025, x_smth, sqrt(v_smth));
ucl_x = norminv(0.975, x_smth, sqrt(v_smth));

certainty = 1 - normcdf(prctile(x_smth, 50) * ones(1, length(x_smth)), x_smth, sqrt(v_smth));

lcl_p = zeros(1, K);
ucl_p = zeros(1, K);

disp('Calculating the pk confidence limits... (this can take time due to the resolution)');
for k = 1:K
    [lcl_p(k), ucl_p(k)] = get_pk_conf_lims(v_smth(k), b0, x_smth(k));
end
disp('Finished calculating the pk confidence limits.');

fs = 4;
t = (0:(K - 1)) / fs;
tr = ((K - 1):(-1):0) / fs;

u_plot = NaN * ones(1, K);
u_plot(pt) = u(pt);

subplot(511);
hold on;
plot(ty, y, 'k', 'linewidth', 1.25);
ylabel({'(a) skin cond.', '(\mu S)'}); 
set(gca,'xticklabel', []); ylim([0 3]);
title('State Estimation with Experimental Data'); xlim([0 ty(end)]);
grid;
yl = ylim;

patch([xp(1), xp(2), xp(2), xp(1)] / fs, [yl(1) yl(1) yl(2) yl(2)], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(2), xp(3), xp(3), xp(2)] / fs, [yl(1) yl(1) yl(2) yl(2)], 'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(3), xp(4), xp(4), xp(3)] / fs, [yl(1) yl(1) yl(2) yl(2)], [1 0.647059 0], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(4), xp(5), xp(5), xp(4)] / fs, [yl(1) yl(1) yl(2) yl(2)], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(5), xp(6), xp(6), xp(5)] / fs, [yl(1) yl(1) yl(2) yl(2)], 'y', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

subplot(512);
stem(t, u_plot, 'fill', 'k', 'markersize', 3);
ylabel('(b) n_{k}, r_{k}'); grid; xlim([0 t(end)]); ylim([0 15]);
yl = ylim; set(gca,'xticklabel', []); 

patch([xp(1), xp(2), xp(2), xp(1)] / fs, [yl(1) yl(1) yl(2) yl(2)], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(2), xp(3), xp(3), xp(2)] / fs, [yl(1) yl(1) yl(2) yl(2)], 'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(3), xp(4), xp(4), xp(3)] / fs, [yl(1) yl(1) yl(2) yl(2)], [1 0.647059 0], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(4), xp(5), xp(5), xp(4)] / fs, [yl(1) yl(1) yl(2) yl(2)], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(5), xp(6), xp(6), xp(5)] / fs, [yl(1) yl(1) yl(2) yl(2)], 'y', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

subplot(513);
hold on;
plot(t, x_smth, 'b', 'linewidth', 1.25);
fill([t, tr], [lcl_x fliplr(ucl_x)], 'c', 'EdgeColor', 'none', 'FaceAlpha', 0.5);
ylabel('(c) state (x_{k})'); ylim([-10 5]);
set(gca,'xticklabel', []); xlim([0 t(end)]);
grid; yl = ylim;

patch([xp(1), xp(2), xp(2), xp(1)] / fs, [yl(1) yl(1) yl(2) yl(2)], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(2), xp(3), xp(3), xp(2)] / fs, [yl(1) yl(1) yl(2) yl(2)], 'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(3), xp(4), xp(4), xp(3)] / fs, [yl(1) yl(1) yl(2) yl(2)], [1 0.647059 0], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(4), xp(5), xp(5), xp(4)] / fs, [yl(1) yl(1) yl(2) yl(2)], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(5), xp(6), xp(6), xp(5)] / fs, [yl(1) yl(1) yl(2) yl(2)], 'y', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

subplot(514);
hold on;
plot(t, p_smth, 'r', 'linewidth', 1.5);
fill([t, tr], [lcl_p fliplr(ucl_p)], [1, 0, (127 / 255)], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
ylim([0 0.15]);
ylabel('(d) probability (p_{k})');
set(gca,'xticklabel', []); xlim([0 t(end)]);
grid; yl = ylim;

patch([xp(1), xp(2), xp(2), xp(1)] / fs, [yl(1) yl(1) yl(2) yl(2)], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(2), xp(3), xp(3), xp(2)] / fs, [yl(1) yl(1) yl(2) yl(2)], 'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(3), xp(4), xp(4), xp(3)] / fs, [yl(1) yl(1) yl(2) yl(2)], [1 0.647059 0], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(4), xp(5), xp(5), xp(4)] / fs, [yl(1) yl(1) yl(2) yl(2)], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
patch([xp(5), xp(6), xp(6), xp(5)] / fs, [yl(1) yl(1) yl(2) yl(2)], 'y', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

subplot(515);
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
ylabel('(d) HAI'); xlabel('time (s)'); xlim([0 t(end)]);

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

function [lcl, ucl] = get_pk_conf_lims(v, b0, x)

    p = (1e-6:1e-6:1);
  
    fp = cumtrapz(p, 1 ./ (sqrt(2 * pi * v) * p .* (1 - p)) .* ...
        exp(((-1) / (2 * v))* (log(p ./ ((1 - p) * exp(b0))) - x) .^ 2));
    
    n = find(fp <= 0.975);
    m = find(fp < 0.025);
    
    ucl = p(n(end));
    lcl = p(m(end));
end


% This code implements the EM algorithm for the state-space model described
% in the following publication.
%
% Wickramasuriya, D. S., & Faghih, R. T. (2019, July). A novel filter for 
% tracking real-world cognitive stress using multi-time-scale point process 
% observations. In 2019 41st Annual International Conference of the IEEE 
% Engineering in Medicine and Biology Society (EMBC) (pp. 599-602). IEEE.
%
% Note: This code takes considerable time to run. Just for convenience and 
% purposes of illustration, the two values on lines 71 and 72 have been adjusted 
% so that they are very close to where the algorithm converges. In general, 
% consider running this code on a cluster. In addition, the code needs to be 
% run for multiple values of eta (line 74) and the best one selected based on the 
% log-likelihood term (line 193).

close all;
clear;
clc;

load('expm_data_one_bin_one_spk.mat');

fs = 4;
delta = 0.005;

min_peak_height = 0.1;
min_peak_promn = 0.1;
min_peak_dist = fs;

ph = s.ph;
tn = s.tn;

rpeaks = s.rpeaks;
ul = s.ul;
w = s.w;
theta = s.theta;

[pks, locs] = findpeaks(ph, 'MinPeakHeight', min_peak_height, 'MinPeakProminence', ...
    min_peak_promn, 'MinPeakDistance', min_peak_dist);

n = zeros(1, length(ph));
n(locs) = 1;

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

tpc = 289;  % total (SCR) peak count
tsl = 34182;  % total signal length

base_prob = tpc / tsl;
b0 = log(base_prob / (1 - base_prob));
tol = 5e-8; % convergence criteria

A = zeros(1, K);
W = zeros(1, K);
CW = zeros(1, K);
C = zeros(1, K);

x_smth(1) = 0.44201528159733;
ve(1) = 1.24111644606324e-4;

eta = -0.00004;

exception_counter = 0;

for m = 1:M
    
    for k = 1:K
        
        if (k == 1)
            x_pred(k) = x_smth(1);
            v_pred(k) = ve(m) + ve(m);
        else
            x_pred(k) = x_updt(k - 1);
            v_pred(k) = v_updt(k - 1) + ve(m);
        end
        
        C(k) = v_pred(k);

        try     % numerical issues can occur due to the integrals
            [x_updt(k), H2] = get_posterior_mode(x_pred(k), C(k), b0, n(k), rpeaks(k, :), ul(k, :), delta, s.w(k, :, :), theta', eta);
            p_updt(k) = 1 / (1 + exp((-1) * (b0 + x_updt(k))));
            v_updt(k) = 1 / ((1 / v_pred(k)) + p_updt(k) * (1 - p_updt(k)) - H2);
        catch         
            exception_counter = exception_counter + 1;
            x_updt(k) = x_pred(k);
            v_updt(k) = v_pred(k);
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

p_updt = 1 ./ (1 + exp((-1) * (b0 + x_updt)));
p_smth = 1 ./ (1 + exp((-1) * (b0 + x_smth)));

t = (0:(K - 1)) / (fs * 60);
tr = ((K - 1):(-1):0) / (fs * 60);

lcl_x = norminv(0.025, x_smth, sqrt(v_smth));
ucl_x = norminv(0.975, x_smth, sqrt(v_smth));

lcl_p = zeros(1, K);
ucl_p = zeros(1, K);

for k = 1:K
    [lcl_p(k), ucl_p(k)] = get_pk_conf_lims(v_smth(k), b0, x_smth(k));
end

certainty = get_certainty_curve(v_smth, b0, x_smth, base_prob);

lambda = zeros(K, 50);
mean_rr = zeros(K, 50);

for k = 1:K
    for j = 1:50
        w = [squeeze(s.w(k, j, :))' [eta x_smth(k)]];
        if (f(theta', ul(k, j), w) > 1e-18)
           lambda(k, j) = fetch_lambda(theta', ul(k, j), w);       
        end
        mean_rr(k, j) =  mu(theta', w);
    end
end

lambda_start_index = find(reshape(rpeaks', 1, numel(rpeaks)), 1);
lambda = reshape(lambda', 1, numel(lambda));
get_ks_plot(find(reshape(rpeaks', 1, numel(rpeaks))) * delta, lambda(lambda_start_index:end), delta, 1);

ll = get_log_likelihood(eta, rpeaks, ul, delta, s.w, theta', x_smth, v_smth);
ll_final = sum(nansum(ll));
mean_rr = reshape(mean_rr', 1, numel(mean_rr));

rri = diff(s.rpeak_locs);
rr_times = s.rpeak_locs(2:end) / 60;

state_ylim = [(min(lcl_x) - 0.1) (max(ucl_x) + 0.1)];
rr_ylim = [(prctile(rri, 1) - 0.05) (prctile(rri, 99) + 0.05)];
prob_ylim = [(min(lcl_p) - 0.0005) (max(ucl_p(3:end)) + 0.0005)];

figure;
subplot(611);
hold on;
plot(t, s.x, 'color', [(102 / 255) 0 (204 / 255)]); grid; 
set(gca,'xticklabel', []);
ylabel('(a) z_{k}'); xlim([0 t(end)]); ylim([4 22]); title('State Estimation with Experimental Data');

subplot(612);
n_plot = NaN * ones(1, K);
n_plot(n > 0) = 1;
stem(t, n_plot, 'fill', 'color', [1, 0, 1], 'markersize', 2);
xlim([0 t(end)]); ylim([0 1.25]);
set(gca,'xticklabel', []);
ylabel('(b) n_{k}'); grid;

subplot(613);
hold on;
plot(t, x_smth, 'b', 'linewidth', 1.25); grid;
set(gca,'xticklabel', []);
fill([t, tr], [lcl_x fliplr(ucl_x)], 'c', 'EdgeColor', 'none', 'FaceAlpha', 0.2);
ylabel('(c) x_{k}'); xlim([0 t(end)]); ylim(state_ylim);

subplot(614);
hold on;
plot(t, p_smth, 'color', [(102 / 255), 0, (51 / 255)], 'linewidth', 1.25); grid;
set(gca,'xticklabel', []);
fill([t, tr], [lcl_p fliplr(ucl_p)], [1, 0, (127 / 255)], 'EdgeColor', 'none', 'FaceAlpha', 0.2);
ylabel('(d) p_{k}'); xlim([0 t(end)]); ylim([0.0012 0.0388]); 
plot([0, t(end)], [base_prob, base_prob], 'k--', 'linewidth', 1.25);

subplot(615);
hold on;
plot(rr_times, rri, 'o', 'col', [1, 0.5, 0.25], ...
    'MarkerFaceColor', [1, 0.5, 0.25], 'MarkerSize', 2); grid;
set(gca,'xticklabel', []);
mu_start_index = round(s.rpeak_locs(2) / delta);
plot(((0:(length(mean_rr(mu_start_index:end)) - 1)) * delta) / 60, mean_rr(mu_start_index:end), 'b');
ylabel('(e) rr_{i}'); xlim([0 t(end)]); ylim(rr_ylim); 

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

plot(t, certainty, 'b', 'linewidth', 1.25); grid; xlim([0 t(end)]);
ylabel('(f) HAI'); xlabel('time (min)'); ylim([0 1]);

function [y, H2] = get_posterior_mode(x_pred, C, b0, n, rpeaks, ul, delta, w_all, theta, eta)

    M = 40;    % maximum iterations
    
    it = zeros(1, M);
    func = zeros(1, M);
    df = zeros(1, M);
    
    it(1) = x_pred;

    for i = 1:(M - 1)
        
        H1 = zeros(1, 50);
        H2 = zeros(1, 50);
        
        for j = 1:50
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

       func(i) = it(i) - x_pred - C * (n - exp(b0 + it(i)) / (1 + exp(b0 + it(i))) + H1);
       df(i) = 1 + C * (exp(b0 + it(i)) / (1 + exp(b0 + it(i))) ^ 2 - H2);
       it(i + 1)  = it(i) - func(i) / df(i);
       
       if abs(it(i + 1) - it(i)) < 1e-14 
           y = it(i + 1);
          return
       end
    end
    
    error('Newton-Raphson failed to converge.');

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
    y = theta(1) + w(1:3) * theta(2:4)' + eta * x;

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
                
                value = nkj * log(delta * lambda) - delta * lambda + ...
                    (d2l_dx2 * (nkj - lambda * delta) / lambda - nkj * (dl_dx ^ 2) / (lambda ^ 2)) * v(k) * 0.5;
                
                if ~isnan(value)
                    y(k, j) = value;
                end
            end
            
        end
    end 
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

function certainty = get_certainty_curve(vK, mu, xK, chance_prob)

    p = (1e-4:1e-4:1);
    [~, i] = min(abs(p - chance_prob));
    certainty = zeros(1, length(vK));
  
    for j = 1:length(vK)
        fp = cumtrapz(p, 1 ./ (sqrt(2 * pi * vK(j)) * p .* (1 - p)) .* ...
            exp(((-1) / (2 * vK(j)))* (log(p ./ ((1 - p) * exp(mu))) - xK(j)) .^ 2));
        certainty(1, j) = 1 - fp(i);
    end
end

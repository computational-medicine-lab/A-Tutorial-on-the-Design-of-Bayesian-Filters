
% This version of the code implements overifitting control. Change the
% value of lambda (line 74), the initial guess for vs (line 72) and the
% stopping threshold (line 130) to control the degree of overfitting.

close all;
clear;
clc;

%% sample data

load('expm_data_one_mpp_one_cont.mat');

subj = 1;
T = 1450;
n = zeros(1, T);
r = zeros(1, T);

pt = find(u > 0);
n(pt) = 1;
r(pt) = u(pt);
s = y;

base_prob = sum(n) / length(n);
pt = find(n > 0);

%% parameters

M = 1e6;    % maximum iterations
m = 1;
tol = 1e-8; % convergence criteria

b0 = zeros(1, M);
b1 = zeros(1, M);

r0 = zeros(1, M);   % continuous model
r1 = zeros(1, M);
vr = zeros(1, M);   % continuous model noise variance (1)

s0 = zeros(1, M);   % continuous model
s1 = zeros(1, M);
vs = zeros(1, M);   % continuous model noise variance (2)

ve = zeros(1, M);   % process noise variance
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
r0(1) = prctile(r(pt), 50);
r1(1)  = 0.5;
s0(1) = s(1);
s1(1) = 1;
vr(1) = 0.05;
vs(1) = 1 * var(s); % 1 * var(s)
ve(1) = 0.05;
lambda = 0.01;  % 0.01

%% main function

for m = 1:M
    
    for k = 1:K
        
        if (k == 1)
            x_pred(k) = x_smth(1);
            v_pred(k) = ve(m) + ve(m);
        else
            x_pred(k) = x_updt(k - 1);
            v_pred(k) = v_updt(k - 1) + ve(m);
        end
        
        x_updt(k) = get_posterior_mode(x_pred(k), v_pred(k), r(k), r0(m), r1(m), b0(m), b1(m), vr(m), n(k), s(k), s0(m), s1(m), vs(m)); 
        p_updt(k) = 1 / (1 + exp((-1) * (b0(m) + b1(m) * x_updt(k))));
        
        if (n(k) == 0)
            v_updt(k) = 1 / ((1 / v_pred(k)) + ((s1(m) ^ 2) / vs(m)) + (b1(m) ^ 2) * p_updt(k) * (1 - p_updt(k)));
        elseif (n(k) == 1)
            v_updt(k) = 1 / ((1 / v_pred(k)) + ((r1(m) ^ 2) / vr(m)) + ((s1(m) ^ 2) / vs(m)) + (b1(m) ^ 2) * p_updt(k) * (1 - p_updt(k)));
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

    R = get_linear_parameters_for_mpp(x_smth, W, r, pt);
    S = get_linear_parameters(x_smth, W, s, K); 
    
    prev = [r0(m) r1(m) ve(m) vr(m) s0(m) s1(m) vs(m) b0(m) b1(m)];
    
    ve(m + 1) = (sum(W(2:end)) + sum(W(1:(end - 1))) - 2 * sum(CW)) / K;
    
    bb = fsolve(@(b) binary_parameter_derivatives(b, n, x_smth, v_smth), [-5 1], optimset('Display','off'));

    b0(m + 1) = bb(1);
    b1(m + 1) = bb(2); 
    
    r0(m + 1) = R(1, 1);
    r1(m + 1) = R(2, 1);
    vr(m + 1) = get_maximum_variance_for_mpp(r, r0(m + 1), r1(m + 1), W, x_smth, pt);

    if ((vs(m) + lambda * (get_maximum_variance(s, s0(m), s1(m), W, x_smth, K) - vs(m))) > 0.75 * var(s))  % EM algorithm intentionally modified slightly for overfitting control
        s0(m + 1) = s0(m) + lambda * (S(1, 1) - s0(m));
        s1(m + 1) = s1(m) + lambda * (S(2, 1) - s1(m));    
        vs(m + 1) = vs(m) + lambda * (get_maximum_variance(s, s0(m), s1(m), W, x_smth, K) - vs(m));       
    else
        s0(m + 1) = s0(m);
        s1(m + 1) = s1(m);
        vs(m + 1) = vs(m);
    end 
    
    next = [r0(m + 1) r1(m + 1) ve(m + 1) vr(m + 1) s0(m + 1) s1(m + 1) vs(m + 1) b0(m + 1) b1(m + 1)];
    
    mean_dev = mean(abs(next - prev));
    
    if (b1(m + 1) < 0) || (r1(m + 1) < 0)   % if this happens with experimental data
        fprintf('Iterations halted at at m = %d\n\n', m);
        break;
    end

    if mean_dev < tol    
        fprintf('Converged at m = %d\n\n', m);
        break;
    else
        fprintf('m = %d\nr0 = %.18f\nr1 = %.18f\nvr = %.18f\n\ns0 = %.18f\ns1 = %.18f\nvs = %.18f\n\nb0 = %.18f\nb1 = %.18f\n\nve = %.18f\n\n', ...
        m + 1, r0(m + 1), r1(m + 1), vr(m + 1), s0(m + 1), s1(m + 1), vs(m + 1), b0(m + 1), b1(m + 1), ve(m + 1));

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

lcl_x = norminv(0.025, x_smth, sqrt(v_smth));
ucl_x = norminv(0.975, x_smth, sqrt(v_smth));

lcl_p = zeros(1, K);
ucl_p = zeros(1, K);

for k = 1:K
    [lcl_p(k), ucl_p(k)] = get_pk_conf_lims(v_smth(k), b0(m), b1(m), x_smth(k));
end

r_plot = NaN * ones(1, K);
r_plot(pt) = r(pt);

%% plot graphs

t = (1:K);
tr = (K:(-1):1);
xtick_pos = 1:(4 * 60):1450;
xtick_labels = {'9 AM', '1 PM', '5 PM', '9 PM', '1 AM', '5 AM', '9 AM'};

subplot(411);
hold on;
stem(t, r_plot, 'fill', 'color', 'b', 'markersize', 4);
plot(t, r_smth, 'r-.', 'linewidth', 1.25);
ylabel('(a) n_{k}, r_{k}'); ylim([-inf (max([r_plot, r_smth]) + 2.5)]);
grid; xlim([0, K]); set(gca, 'xtick', xtick_pos);
set(gca, 'xticklabel', []);
title('State Estimation with Experimental Data'); 

subplot(412);
hold on;
plot(t, s, 'color', [1 (128 / 255) 0], 'linewidth', 1.25); grid;
plot(t, s_smth, 'r-.', 'linewidth', 1.25);
ylim([0 (max([s, s_smth]) + 2.5)]);
ylabel('(b) s_{k}'); set(gca, 'xtick', xtick_pos);
xlim([0, K]); set(gca, 'xticklabel', []);

subplot(413);
hold on;
col = [0 (176 / 255) (80 / 255)];
fill([t, tr], [lcl_p fliplr(ucl_p)], [(54 / 255) (208 / 255) (80 / 255)], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(t, p_smth, 'color', [(54 / 255) (150 / 255) (80 / 255)], 'linewidth', 1.25); grid;
ylabel('(c) p_{k}'); set(gca, 'xtick', xtick_pos); ylim([0 (max(ucl_p) + 0.0075)]);
xlim([0, K]); set(gca, 'xticklabel', []);

subplot(414);
hold on; 
fill([t, tr], [lcl_x fliplr(ucl_x)], [(102 / 255) 0 (204 / 255)], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
plot(t, x_smth, 'color', [(102 / 255) 0 (150 / 255)], 'linewidth', 1.25);
grid; xlim([0, K]); ylim([(min(lcl_x) - 1) (max(ucl_x) + 1)]);
set(gca, 'xtick', xtick_pos);
set(gca, 'xticklabel', xtick_labels);
ylabel('(d) x_{k}'); xlabel('time');

function y = get_posterior_mode(x_pred, v_pred, r, r0, r1, b0, b1, vr, n, s, s0, s1, vs)

    M = 200;    % maximum iterations
    
    it = zeros(1, M);
    f = zeros(1, M);
    df = zeros(1, M);
    
    it(1) = x_pred;

    for i = 1:(M - 1)
        
        if (n == 0)
            C = v_pred / ((s1 ^ 2) * v_pred + vs);
            f(i) = it(i) - x_pred - C * (s1 * (s - s0 - s1 * x_pred) + vs * b1 * (n - (1 / (1 + exp((-1) * (b0 + b1 * it(i))))))); 
            df(i) = 1 + C * vs * (b1 ^ 2) * exp(b0 + b1 * it(i)) / ((1 + exp(b0 + b1 * it(i))) ^ 2);            
        elseif (n == 1)
            C = v_pred / (vr * vs + v_pred * ((r1 ^ 2) * vs + (s1 ^ 2) * vr));
            f(i) = it(i) - x_pred - C * (r1 * vs * (r - r0 - r1 * x_pred) + s1 * vr * (s - s0 - s1 * x_pred) + ...
                vr * vs * b1 * (n - (1 / (1 + exp((-1) * (b0 + b1 * it(i))))))); 
            df(i) = 1 + C * vr * vs * (b1 ^ 2) * exp(b0 + b1 * it(i)) / ((1 + exp(b0 + b1 * it(i))) ^ 2);
        end
        
        it(i + 1) = it(i) - f(i) / df(i);
        
        if abs(it(i + 1) - it(i)) < 1e-14
           y = it(i + 1);
           return;
        end
    end
    
    error('Newton-Raphson failed to converge.');

end

function [lcl, ucl] = get_pk_conf_lims(v, b0, b1, x)

    p = (1e-4:1e-4:1);
  
    fp = cumtrapz(p, 1 ./ (sqrt(2 * pi * v) * b1 * p .* (1 - p)) .* ...
        exp(((-1) / (2 * v))* ((1 / b1) * log(p ./ ((1 - p) * exp(b0))) - x) .^ 2));
    
    n = find(fp <= 0.975);
    m = find(fp < 0.025);
    
    ucl = p(n(end));
    lcl = p(m(end));
end

function y = get_maximum_variance_for_mpp(z, r0, r1, W, x_smth, pt)

    x_smth = x_smth(pt);
    W = W(pt);
    z = z(pt);
    K = length(pt);
    
    y = (z * z' + K * (r0 ^ 2) + (r1 ^ 2) * sum(W) ...
                - 2 * r0 * sum(z) - 2 * r1 * dot(x_smth, z) + 2 * r0 * r1 * sum(x_smth)) / K;      
end

function y = get_maximum_variance(z, r0, r1, W, x_smth, K)

    y = (z * z' + K * (r0 ^ 2) + (r1 ^ 2) * sum(W) ...
                - 2 * r0 * sum(z) - 2 * r1 * dot(x_smth, z) + 2 * r0 * r1 * sum(x_smth)) / K;          
end

function y = get_linear_parameters_for_mpp(x_smth, W, z, pt)

    x_smth = x_smth(pt);
    W = W(pt);
    z = z(pt);
    K = length(pt);
    
    y = [K sum(x_smth); sum(x_smth) sum(W)] \ [sum(z); sum(z .* x_smth)];
    
end

function y = get_linear_parameters(x_smth, W, z, K)

    y = [K sum(x_smth); sum(x_smth) sum(W)] \ [sum(z); sum(z .* x_smth)];
    
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


close all;
clear;
clc;

ndays = 5;
T = 1440;
N = ndays * T;
t = (1:N);

load('expm_data_one_bin_two_cont_circadian.mat');

std_r = std(r);
std_s = std(s); 

r = r / std_r;
s = s / std_s;

base_prob = sum(n) / length(n);

M = 2e6;
ve = zeros(1, M);   % process noise variance
rho = zeros(1, M);
b0 = zeros(1, M);
b1 = zeros(1, M);

r0 = zeros(1, M);
r1 = zeros(1, M);
vr = zeros(1, M);

s0 = zeros(1, M);
s1 = zeros(1, M);
vs = zeros(1, M);

K = length(n);

x_pred = zeros(1, K);
v_pred = zeros(1, K);

x_updt = zeros(1, K);
v_updt = zeros(1, K);

x_smth = zeros(1, K);
v_smth = zeros(1, K);

p_updt = zeros(1, K);

tol = 1e-8; % convergence criteria

A = zeros(1, K);
W = zeros(1, K);
CW = zeros(1, K);
C = zeros(1, K);

ve(1) = 0.005;
rho(1) = 0.98;

b0(1) = log(base_prob / (1 - base_prob));
b1(1) = 0.9;

r0(1) = r(1);
r1(1)  = 1;
vr(1) = 0.005;

s0(1) = s(1);
s1(1) = 1;
vs(1) = 0.005;

for m = 1:M
    
    for k = 1:K
        
        if (k == 1)
            x_pred(k) = x_smth(1) + I(k);
            v_pred(k) = ve(m) + ve(m);
        else
            x_pred(k) = rho(m) * x_updt(k - 1) + I(k);
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
        
        rho(m + 1) = sum(CW) / sum(W(1:end - 1));
        
        next_ve = (sum(W(2:end)) + (rho(m + 1) ^ 2) * sum(W(1:(end - 1))) - 2 * rho(m + 1) * sum(CW) - ...
            2 * (I(2:end) * x_smth(2:end)') + 2 * rho(m + 1) * (I(2:end) * x_smth(1:(end - 1))') + ...
            (I * I')) / K;
        
        if (next_ve > 0)    % check - in case this happens with experimental data
            ve(m + 1) =  next_ve;
        else
            ve(m + 1) = ve(m);
        end
        
        bb = fsolve(@(b) binary_parameter_derivatives(b, n, x_smth, v_smth), [-5 1], optimset('Display','off'));
        
        if (bb(2) > 0)    % check - in case this happens with experimental data 
            b0(m + 1) = bb(1);
            b1(m + 1) = bb(2); 
        else
            b0(m + 1) = b0(m);
            b1(m + 1) = b1(m);
        end

        a = fminsearch(@(a) circadian_parameters(a, rho(m + 1), x_smth, t, T), a, optimset('Display', 'off'));
        I = rhythm(a, T, t);
        
        R = get_linear_parameters(x_smth, W, r, K);
        S = get_linear_parameters(x_smth, W, s, K);
        
        next_vr = get_continuous_variable_variance_update(r, R(1, 1), R(2, 1), W, x_smth, K);
        next_vs = get_continuous_variable_variance_update(s, S(1, 1), S(2, 1), W, x_smth, K);
        
        if (abs(next_vr - next_vs) > 0.01)  % overfitting control with experimental data
            r0(m + 1) = r0(m);
            r1(m + 1) = r1(m);
            
            s0(m + 1) = s0(m);
            s1(m + 1) = s1(m);
            
            vr(m + 1) = vr(m);
            vs(m + 1) = vs(m);
        else        
            r0(m + 1) = R(1, 1);
            r1(m + 1) = R(2, 1);
        
            s0(m + 1) = S(1, 1);
            s1(m + 1) = S(2, 1);
        
            vr(m + 1) = next_vr;
            vs(m + 1) = next_vs;
        end
        
        mean_dev = mean(abs([ve(m + 1) rho(m + 1) r0(m + 1) r1(m + 1) vr(m + 1) s0(m + 1) s1(m + 1) vs(m + 1) b1(m + 1) b0(m + 1)] - ...
            [ve(m) rho(m) r0(m) r1(m) vr(m) s0(m) s1(m) vs(m) b1(m) b0(m)]));        
            
        if mean_dev < tol 
            fprintf('m = %d\nx0 = %.18f\nve = %.18f\nrho = %.18f\n\nr0 = %.18f\nr1 = %.18f\nvr = %.18f\ns0 = %.18f\ns1 = %.18f\nvs = %.18f\n\nb0 = %.18f\nb1 = %.18f\n\n', ...
                m, x_smth(1), ve(m), rho(m), r0(m), r1(m), vr(m), s0(m), s1(m), vs(m), b0(m), b1(m));
            fprintf('Converged at m = %d\n\n', m);          
            break;
        else
            fprintf('m = %d\nx0 = %.18f\nve = %.18f\nrho = %.18f\n\nr0 = %.18f\nr1 = %.18f\nvr = %.18f\n\ns0 = %.18f\ns1 = %.18f\nvs = %.18f\n\nb0 = %.18f\nb1 = %.18f\n\n', m, ...
                x_smth(1), ve(m + 1), rho(m + 1), r0(m + 1), r1(m + 1), vr(m + 1), s0(m + 1), s1(m + 1), vs(m + 1), b0(m + 1), b1(m + 1));
        
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

p_smth = 1 ./ (1 + exp((-1) * (b0(m) + b1(m) * x_smth)));  
r_smth = (r0(m) + r1(m) * x_smth) * std_r;
s_smth = (s0(m) + s1(m) * x_smth) * std_s;

index = (0:(K - 1));
t_index = index / (60 * 24);
r_index = ((K - 1):(-1):0); % reverse index
transp = 0.3;
 
subplot(611);
hold on;
plot(t_index, y, 'color', [(102 / 255) 0 (204 / 255)]); grid;
ylabel('(a) z_{k}');
title('State Estimation with Experimental Data');
xlim([0 t_index(end)]); ylim([0 (1.1 * max(y))]);
set(gca,'xticklabel', []);

subplot(612);
n_plot = NaN * ones(1, K);
n_plot(n > 0) = 1;
stem(t_index, n_plot, 'fill', 'color', [1, 69 / 255, 0], 'markersize', 2);
xlim([0 t_index(end)]); ylim([0 1.25]);
set(gca,'xticklabel', []);
ylabel('(b) n_{k}'); grid;

subplot(613);
hold on;
plot(t_index, p_smth, 'r', 'linewidth', 1.5); ylim([(0.98 * min(p_smth)) (1.08 * max(p_smth))]);
ylabel('(c) p_{k}'); grid;
xlim([0 t_index(end)]);
set(gca,'xticklabel', []);

subplot(614);
hold on;
plot(t_index, r_smth, '--', 'color', [0 0.3 0], 'linewidth', 2);
plot(t_index, r * std_r, 'color', [0 0.9 0]); grid;
xlim([0 t_index(end)]); ylabel('(d) r_{k}');
set(gca,'xticklabel', []);

subplot(615);
hold on;
plot(t_index, s_smth, '--', 'color', [0.5 (25 / 255) (66 / 255)], 'linewidth', 2);
plot(t_index, s * std_s, 'color', [1 0.5 (179 / 255)]); grid;
xlim([0 t_index(end)]); ylabel('(e) s_{k}'); 
set(gca,'xticklabel', []);

subplot(616);
hold on;
plot(t_index, x_smth, 'b', 'linewidth', 1.5);  
ylabel('(f) x_{k}');
xlim([0 t_index(end)]); ylim([(min(x_smth) - 1) (max(x_smth) + 1)]);
grid; 

xticks(0:0.5:4.5); xticklabels({'0000', '1200', '0000', '1200', '0000', '1200', '0000', '1200', '0000', '1200'});
xlabel('time (24h clock)'); 


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

function y = get_linear_parameters(x_smth, W, z, K)
    
    y = [K sum(x_smth); sum(x_smth) sum(W)] \ [sum(z); sum(z .* x_smth)];
    
end

function y = get_continuous_variable_variance_update(z, r0, r1, W, x_smth, K)

    y = (z * z' + K * (r0 ^ 2) + (r1 ^ 2) * sum(W) ...
            - 2 * r0 * sum(z) - 2 * r1 * dot(x_smth, z) + 2 * r0 * r1 * sum(x_smth)) / K;

end

function y = circadian_parameters(a, rho, x_smth, t, T)

    I = rhythm(a, T, t);
    y = (I * I') - 2 * (I(2:end) * x_smth(2:end)') + 2 * rho * (I(2:end) * x_smth(1:(end - 1))');
end

function y = rhythm(a, T, t)   % the a0 is ignored
    y = 0 + a(2) * sin(2 * pi * t / T) + a(3) * cos(2 * pi * t / T) + ...
        a(4) * sin(2 * pi * t / (T / 2)) + a(5) * cos(2 * pi * t / (T / 2));    
end
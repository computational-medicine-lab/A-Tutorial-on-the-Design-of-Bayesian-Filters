function [KSdistance,Z,T] = get_ks_plot(EKGR, L, delta, plt)
% function [KSdistance,Z,T] = ks_plot(EKGR, L, delta, plt)
%
% Evaluate the KS-statistics for a set of observed events.
%
% [KSdistance,Z,T] = ks_plot(EKGR, L, delta, plt)
% where:
%    EKGR is a vector with the time of occurrence of the events
%    L is a vector containing the values of the hazard-rate function in time
%    delta is the sampling time of the lambda function
%    plt, if nonzero a KS-plot is produced (this is the default)
%
%    KSdistance is the maximum KS distance
%    Z is the transformed integral of lambda (which is tested for uniform
%            distribution)
%    T is the integral of lambda evaluated at each event
%
%
% Copyright (C) Luca Citi and Riccardo Barbieri, 2010-2011.
% All Rights Reserved. See LICENSE.TXT for license details.
% {lciti,barbieri}@neurostat.mit.edu
% http://users.neurostat.mit.edu/barbieri/pphrv

if nargin < 4
    plt = 1;
end

first = find(~isnan(L), 1);

EKGR = EKGR(:) - EKGR(1); % times are relative to EKGR(1) which is set to 0
L(ceil(EKGR(end) / delta)+1 : end) = [];

lastRi = find(EKGR > first * delta, 1) - 1;

T = NaN(size(EKGR));

intL = NaN;
pr = NaN;

for j = first:length(L)%j = first:length(L)+1
    time = (j-1) * delta;    
    event = EKGR(lastRi + 1) <= time; % whether an event happened in ((j-1)*delta,j*delta]
    if event
        lastRi = lastRi + 1;
        % consider the time before the R event for the previous spike
        dt = EKGR(lastRi) - (time - delta);
        intL = intL + dt * pr;
        T(lastRi) = intL;
        intL = 0;
    else
        intL = intL + delta * L(j);
    end
    % consider the time after the R event for the next spike
    if j <= length(L)
        pr = L(j);
    end
end

Z = 1 - exp(-T);
Z = Z(~isnan(Z));

ordered = sort(Z);
ordered = ordered(:)';
d = length(ordered);
lin = linspace(0,1,d);
lu = linspace(1.36/sqrt(d), 1+1.36/sqrt(d), d);
ll = linspace(-1.36/sqrt(d), 1-1.36/sqrt(d), d);

KSdistance = max(abs(ordered-lin)) / sqrt(2);

if plt

    plot(ordered, lin, 'k', 'linewidth', 3);
    hold on;
    plot(lin, lin, 'r', 'linewidth', 1.5);
    plot(lu, lin, 'b', 'linewidth', 1.5);
    plot(ll, lin, 'b', 'linewidth', 1.5);
    xlim([0 1])
    ylim([0 1])
    grid;
    xlabel('empirical quantiles');
    ylabel('theoretical quantiles');
    title('KS Plot');

end


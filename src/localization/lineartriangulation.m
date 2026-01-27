function [rr_origin2observer, rr_observer2beacons, rhos] = lineartriangulation(loses, rr_origin2beacons)
% Linear Triangulation: calculates a least squares solution for direction vectors
% given line-of-sight vectors (loses) and corresponding beacon positions.
%
% Inputs:
%   LoSes       - 3xN matrix of line-of-sight unit vectors
%   rr_beacons  - 3xN matrix of beacon positions
% 
% Outputs:
%   rr_solutions - 3xN matrix of resulting positions 
%   rhos         - 1xN vector of ranges
% 
% Author: Andrea Pizzetti
% Date:   June 6, 2025

loses = reshape(loses, 3, []);
rr_origin2beacons = reshape(rr_origin2beacons, 3, []);
loses = loses./vecnorm(loses);

[dim, N] = size(rr_origin2beacons);
if N < 2
    error('At least 2 rr_origin2beacons are required');
end

n = size(loses, 2);
c = 0;
H = nan(n*(n-1), n);
b = nan(n*(n-1), 1);
for i = 1:n-1
    for j = i + 1:n

        los_i = loses(:, i);
        los_j = loses(:, j);
        r_i = rr_origin2beacons(:, i);
        r_j = rr_origin2beacons(:, j);

        cgamma_ij = los_i'*los_j;
        
        H_ij = [-1 cgamma_ij; cgamma_ij, -1];
        
        LAMBDA_ij = zeros(2, n);
        LAMBDA_ij(1, i) = 1;
        LAMBDA_ij(2, j) = 1;
        
        b(c + [1:2], :) = [los_i'*(r_j - r_i);
                         los_j'*(r_i - r_j)];

        H(c + [1:2], 1:n) = H_ij*LAMBDA_ij;
        
        c = c + 2;

    end
end


rhos = (H'*H)\(H'*b);
rhos = rhos';

rr_observer2beacons = rhos.*loses;

rr_origin2observer = rr_origin2beacons - rr_observer2beacons;

end


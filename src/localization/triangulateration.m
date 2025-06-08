function rr_origin2observer = triangulateration(loses, ranges, rr_origin2beacons)
% Triangulateration: Estimate observer position by averaging p_i - r_i * d_i
%
% Inputs:
%   loses - 3xN matrix of unit LOS vectors d_i (from observer to beacon)
%   ranges      - 1xN vector of distances r_i
%   rr_origin2beacons     - 3xN matrix of beacon positions p_i
%
% Output:
%   rr_origin2observer - 3x1 estimated observer position

N = size(rr_origin2beacons, 2);
if length(ranges) ~= N || size(loses,2) ~= N
    error('Input sizes mismatch');
end

% Calculate candidate positions from each beacon
candidates = rr_origin2beacons - loses .* ranges;

% Mean position
rr_origin2observer = mean(candidates, 2);

end

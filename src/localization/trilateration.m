function rr_origin2observer = trilateration(ranges, rr_origin2beacons)
% Trilateration: Estimate 3D observer position from ranges to 3 beacons. 
% Returns two possible solutions (3x2 matrix)
%
% Inputs:
%   ranges  - 1x3 vector, distances from observer to each beacon
%   rr_origin2beacons - 3x3 matrix, each column is [x; y; z] of a beacon
%
% Output:
%   rr_origin2observer - 3x2 matrix, each column is a possible solution [x; y; z]
%
% Author: Andrea Pizzetti
% Date: June 6, 2025

if size(rr_origin2beacons,1) ~= 3 || size(rr_origin2beacons,2) ~= 3
    error('beacons must be a 3x3 matrix');
end
if length(ranges) ~= 3
    error('ranges must be a 1x3 vector');
end

P1 = rr_origin2beacons(:,1);
P2 = rr_origin2beacons(:,2);
P3 = rr_origin2beacons(:,3);
r1 = ranges(1);
r2 = ranges(2);
r3 = ranges(3);

% Unit vector from P1 to P2
ex = (P2 - P1) / norm(P2 - P1);
i = dot(ex, P3 - P1);

% Unit vector orthogonal to ex in the plane of P1, P2, P3
temp = P3 - P1 - i*ex;
ey = temp / norm(temp);

% Unit vector orthogonal to plane
ez = cross(ex, ey);

d = norm(P2 - P1);
j = dot(ey, P3 - P1);

% Coordinates in local frame
x = (r1^2 - r2^2 + d^2) / (2*d);
y = (r1^2 - r3^2 + i^2 + j^2) / (2*j) - (i/j)*x;

z_sq = r1^2 - x^2 - y^2;
if z_sq < 0
    warning('No intersection: spheres do not intersect');
    rr_origin2observer = nan(3,2);
    return;
end
z = sqrt(z_sq);

% Two solutions in global coordinates
sol1 = P1 + x*ex + y*ey + z*ez;
sol2 = P1 + x*ex + y*ey - z*ez;

rr_origin2observer = [sol1, sol2];

end

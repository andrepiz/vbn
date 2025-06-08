function uvd = xyz2uvd(K, x, y, z)
% Return the U, V and depth coordinates from the 3D vector 

d = sqrt(x.^2 + y.^2 + z.^2);
d(z<0) = -d(z<0);
xn = x./z;
yn = y./z;
uv1 = K*[xn; yn; ones(1, length(d))];

uvd = [uv1(1,:); uv1(2,:); d];

end
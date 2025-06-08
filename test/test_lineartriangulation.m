mice_install()
set_plot_settings();

%%
filename_metakernel = 'default.tm';


% Constants
mu_sun = 1.32712440018e11*1e9; % [km^3/s^2] solar gravitational parameter
AU = 149597870.707e3;
name_beacons = {'MERCURY','VENUS','EARTH','MARS','JUPITER','SATURN','URANUS','NEPTUNE'};
id_beacons = {'1','2','3','4','5','6','7','8'};
n_beacons = [2:8];

% INPUTS
kep0 = [3.3767*AU; 0.55; deg2rad(24.35); deg2rad(94.41); deg2rad(1.75); deg2rad(10.11)];
% kep0 = [12.3767*AU; 0.2; deg2rad(24.35); deg2rad(94.41); deg2rad(1.75); deg2rad(10.11)];
t0 = '01/01/2025';
nyears = 15;
ltscorrection = 'NONE';
frame = 'J2000';
origin = 'SUN';

%% GROUND TRUTH

time_UTC = datetime(t0,'InputFormat','dd/MM/uuuu') + days(0:2:365*nyears);
kep = propagate_position_keplerian(kep0, time_UTC, mu_sun, filename_metakernel);
rr_observer = kep2car(kep(1, :), kep(2, :), kep(3, :), kep(4, :), kep(5, :), kep(6, :), mu_sun);
[loses, rr_beacons] = extract_loses_bodies_kernel(rr_observer, time_UTC, id_beacons, filename_metakernel, ltscorrection, origin, frame);

%% MEASUREMENTS
rng(0)
sigma_loses = deg2rad(15/3)/3600; % from franzese et al. (planets)
%sigma_loses = deg2rad(30/3)/3600; % from franzese et al. (asteroids)
sigma_rr_beacons = 1/3;   % [km] from franzese et al. (planets)
% sigma_rr_beacons = 100/3;   % [km] from franzese et al. (asteroids)

loses_pert = loses + sigma_loses*randn(size(loses));
rr_beacons_pert = rr_beacons + sigma_rr_beacons*randn(size(rr_beacons));

%% TRIANGULATION
nt = length(time_UTC);
rr_solution = nan(3, nt);
for ix = 1:nt
    for ib = 1:length(n_beacons)
        rr_solutions = lineartriangulation(squeeze(loses_pert(:, ix, 1:n_beacons(ib))), squeeze(rr_beacons_pert(:, ix, 1:n_beacons(ib))));
        rr_solution(:, ix, ib) = median(rr_solutions, 2);
    end
end

%% PLOT

cm1 = colormap('hsv');
cm1 = interp1(linspace(0, 1, size(cm1, 1)), cm1, linspace(0, 1, length(id_beacons)));

% TRAJECTORY _____________________________________________________________
k = 1/AU;
f1 = figure('Name','Orbits','Units','normalized','Position',[0.1 0.1 0.8 0.6]);
hold on
axis equal;
grid on
ax1 = f1.CurrentAxes;
set(f1, 'Color', 'k');       % Set figure background to black
set(ax1, 'Color', 'k');       % Set axes background to black
set(ax1, 'XColor', 'w');      % Set x-axis color to white
set(ax1, 'YColor', 'w');      % Set y-axis color to white
set(ax1, 'ZColor', 'w');      % Set y-axis color to white
set(ax1, 'GridColor', 'w', 'GridAlpha', 0.3);
text(rr_observer(1,1)*k,rr_observer(2,1)*k,rr_observer(3,1)*k,'S/C','Color','w','HorizontalAlignment','right','VerticalAlignment','baseline');
plot3(rr_observer(1,1)*k,rr_observer(2,1)*k,rr_observer(3,1)*k,'wx',"MarkerSize",5);
plot3(rr_observer(1,:)*k,rr_observer(2,:)*k,rr_observer(3,:)*k,'w',"LineWidth",2);
plot3(0,0,0,"*",'Color',[0.8 0.8 0.2],'MarkerSize',10,'LineWidth',10)
for ii = 1: n_beacons(end)
    text(rr_beacons(1,1,ii)*k,rr_beacons(2,1,ii)*k,rr_beacons(3,1,ii)*k,name_beacons{ii},'Color',cm1(ii, :),'HorizontalAlignment','right','VerticalAlignment','baseline')
    plot3(rr_beacons(1,1,ii)*k,rr_beacons(2,1,ii)*k,rr_beacons(3,1,ii)*k,"o",'MarkerSize',5,'Color',cm1(ii, :))
    plot3(rr_beacons(1,:,ii)*k,rr_beacons(2,:,ii)*k,rr_beacons(3,:,ii)*k,"--","LineWidth",1,'Color',cm1(ii, :))
end
xlabel('X [AU]')
ylabel('Y [AU]')
zlabel('Z [AU]')
shading interp;
camlight headlight;
lighting gouraud;
 
% ERROR

cm2 = colormap('default');
cm2 = interp1(linspace(0, 1, size(cm2, 1)), cm2, linspace(0, 1, length(n_beacons)));

figure()
for ix = 1:3
subplot(3,1,ix)
hold on, grid on
for ib = 1:length(n_beacons)
    plot(time_UTC, rr_solution(ix,:,ib)*k,'Color',cm2(ib,:), 'DisplayName',[num2str(n_beacons(ib)),'B'])
end
plot(time_UTC, rr_observer(ix,:)*k,'r--','DisplayName','True')
legend show
xlabel('date')
ylabel('triangulated position [AU]')
end


figure('name','triangulation_error','Units','normalized','Position',[0.1 0.1 0.8 0.4])
hold on, grid on
for ib = 1:length(n_beacons)
    plot(days(time_UTC - time_UTC(1)), 1e-3*vecnorm(rr_solution(:,:,ib) - rr_observer),'DisplayName',[num2str(n_beacons(ib)),'B'])
end
set(gca(), 'YScale','log')
legend show
ylim([1e3 5e6])
xlabel('days')
ylabel('triangulated position error [km]')

%
figure('name','triangulation_error_mean','Units','normalized','Position',[0.1 0.1 0.3 0.4])
hold on, grid on
for ib = 1:length(n_beacons)
    scatter(ib, 1e-3*mean(vecnorm(rr_solution(:,:,ib) - rr_observer)),'o','filled','DisplayName',num2str(n_beacons(ib)))
end
set(gca(), 'YScale','log')
legend show
ylim([1e3 5e6])
xlim([0 11])
xlabel('beacons')
ylabel('triangulated position error [km]')
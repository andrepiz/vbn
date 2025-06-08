clear
close all
clc

cvt_install();
vbn_install();

addpath(genpath("src"))

rng(1)

%% INPUTS
res_px = [2048 1536];
fx = 12e-3/2e-6;
fy = 12e-3/2e-6;
cu = res_px(1)/2;
cv = res_px(2)/2;
K = [fx, 0, cu; 
     0, fy, cv;
     0,  0,  1];

t0 = 10;
tend = 20;
fps = 15;
nt = (tend-t0)*fps; % time vector length
nsp = 9; % propagator state
nsf = 9; % filter state dimension
nzf = 3; % filter measurement dimension

x0 = [-1; 1; 3; 1.5; -1.4; 4; 0.5; -0.6; 4]; % assume constant acceleration
sigma_pos0 = [4; 5; 9];
sigma_vel0 = [1; 2; 4];
sigma_acc0 = [0.5; 0.5; 2];
sigma_px = 1; % [px]
sigma_d = 0.05; % [%] error on relative range
sigma_dt = 0.01; % [%] error on timetag of the measurement

dt0 = 1/fps;
P0 = diag([sigma_pos0.^2; sigma_vel0.^2; sigma_acc0.^2]);

%% RUN

% Initialize propagator
prop = propagator.pointUniformAcc();

% Reference time vector
tTrue = linspace(t0, tend, nt);

% Reference state
xTrue = zeros(nsp, nt);
dtTrue = nan(1, nt);
dtTrue(2:end) = diff(tTrue);
xTrue(:, 1) = x0;
prop.x = x0;
for ix = 2:nt
    prop.propagate(dtTrue(ix))
    xTrue(:, ix) = prop.x;
end

% Measurement
posTrue = xTrue(1:nzf, :);
uvdTrue = xyz2uvd(K, posTrue(1,:), posTrue(2,:), posTrue(3,:));
uvTrue = uvdTrue([1 2], :);
dTrue = uvdTrue(3, :);
ixs_outfov = all(uvTrue < 0 | uvTrue > [res_px(1); res_px(2)], 1) | dTrue < 0;
noutfov = sum(ixs_outfov);
uvMeas = uvTrue + sigma_px.*randn(size(uvTrue));
dMeas = dTrue.*(1 + sigma_d.*randn(size(dTrue)));
uvMeas(:, ixs_outfov) = repmat([nan; nan], 1, noutfov);
dMeas(ixs_outfov) = nan*ones(1, noutfov);
posMeas = uvd2xyz(K, uvMeas(1,:), uvMeas(2,:), dMeas);
dtMeas = dtTrue.*(1 + sigma_dt.*randn(size(tTrue)));
t0Meas = t0*(1 + sigma_dt.*randn(1));
tMeas = t0Meas + [0, cumsum(dtMeas(2:end))];;

% Estimation
xEst = zeros(nsf, nt);
PEst = nan(nsf, nt);
xEst(1:3, 1) = xTrue(1:3,1) + sigma_pos0.*randn(3,1);
xEst(4:6, 1) = xTrue(4:6,1) + sigma_vel0.*randn(3,1);
xEst(7:9, 1) = xTrue(7:9,1) + sigma_acc0.*randn(3,1);
PEst(1:nsf,1) = diag(P0);

flt = ekf.xPosVelAcc_zUVD_dtVariable(xEst(:, 1), P0, dt0, K);
for ix = 2:nt
    
    % Timestep extraction
    dt = tMeas(ix) - tMeas(ix-1);

    % Measurement 
    z = [uvMeas(:, ix); dMeas(:, ix)];
    flagMeasNull = all(isnan(z));

    if ~flagMeasNull
        % predict and update

        R_tuning = 0.001*[sigma_px; sigma_px; sigma_d*dMeas(:, ix)].^2;
        flt.R_scaler = R_tuning;
        flt.update_R;

        flt.predict_and_update(z, dt);    
    else
        % predict only
        flt.predict_only(dt)
    end

    dtMeas(ix) = dt;
    xEst(:, ix) = flt.x;
    PEst(:,ix) = diag(flt.P(1:nsf, 1:nsf));

end

%% POST-PRO

%---state
figure('name','time'), grid on, hold on
plot(tTrue, 1e3*dtTrue, 'g','LineWidth',2)
plot(tTrue, 1e3*dtMeas, 'c--','LineWidth',2)
legend('True','Meas')
xlabel('Time [s]')
ylabel('Time Step [ms]')

figure('name','position'), 
txt_xyz = {'x','y','z'};
for ix = 1:3
    subplot(3,1,ix),grid on, hold on
    plot(tTrue, xTrue(ix,:), 'g','LineWidth',2,'DisplayName',[txt_xyz{ix},'True'])
    plot(tTrue, posMeas(ix,:), 'bo','LineWidth',2,'DisplayName',[txt_xyz{ix},'Meas'])
    plot(tTrue, xEst(ix,:), 'c--','LineWidth',2,'DisplayName',[txt_xyz{ix},'Est'])
    xlabel('Time [s]')
    ylabel('[m]')
    legend show
end

figure('name','velocity'), 
for ix = 1:3
    subplot(3,1,ix),grid on, hold on
    plot(tTrue, xTrue(3+ix,:), 'g','LineWidth',2,'DisplayName',['v',txt_xyz{ix},'True'])
    plot(tTrue, xEst(3+ix,:), 'c--','LineWidth',2,'DisplayName',['v',txt_xyz{ix},'Est'])
    xlabel('Time [s]')
    ylabel('[m/s]')
    legend show
end

figure('name','acceleration'), 
for ix = 1:3
    subplot(3,1,ix),grid on, hold on
    plot(tTrue, xTrue(6+ix,:), 'g','LineWidth',2,'DisplayName',['a',txt_xyz{ix},'True'])
    plot(tTrue, xEst(6+ix,:), 'c--','LineWidth',2,'DisplayName',['a',txt_xyz{ix},'Est'])
    xlabel('Time [s]')
    ylabel('[m/s^2]')
    legend show
end

%---errors
figure('name','position_error'), 
txt_xyz = {'x','y','z'};
for ix = 1:3
    subplot(3,1,ix),grid on, hold on
    plot(tTrue, xTrue(ix,:) - posMeas(ix,:), 'b','LineWidth',2,'DisplayName',[txt_xyz{ix},'Meas Err'])
    plot(tTrue, xTrue(ix,:) - xEst(ix,:), 'c','LineWidth',2,'DisplayName',[txt_xyz{ix},'Est Err'])
    plot(tTrue, sqrt(PEst(ix,:)*3), 'r','LineWidth',1,'DisplayName',['P',txt_xyz{ix},' 3\sigma'])
    plot(tTrue, -sqrt(PEst(ix,:)*3), 'r','LineWidth',1,'HandleVisibility','off')
    xlabel('Time [s]')
    ylabel('[m]')
    legend show
end

figure('name','velocity_error'), 
for ix = 1:3
    subplot(3,1,ix),grid on, hold on
    plot(tTrue, xTrue(3+ix,:) - xEst(3+ix,:), 'g','LineWidth',2,'DisplayName',['v',txt_xyz{ix},'Est Err'])
    plot(tTrue, sqrt(PEst(3+ix,:)*3), 'r','LineWidth',1,'DisplayName',['P',txt_xyz{ix},' 3\sigma'])
    plot(tTrue, -sqrt(PEst(3+ix,:)*3), 'r','LineWidth',1,'HandleVisibility','off')
    xlabel('Time [s]')
    ylabel('[m/s]')
    legend show
end

figure('name','acceleration_error'), 
for ix = 1:3
    subplot(3,1,ix),grid on, hold on
    plot(tTrue, xTrue(6+ix,:) - xEst(6+ix,:), 'g','LineWidth',2,'DisplayName',['a',txt_xyz{ix},'Est Err'])
    plot(tTrue, sqrt(PEst(6+ix,:)*3), 'r','LineWidth',1,'DisplayName',['P',txt_xyz{ix},' 3\sigma'])
    plot(tTrue, -sqrt(PEst(6+ix,:)*3), 'r','LineWidth',1,'HandleVisibility','off')
    xlabel('Time [s]')
    ylabel('[m/s^2]')
    legend show
end
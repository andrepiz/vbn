clear
close all
clc

cvt_install();

addpath(genpath("src"))
addpath(genpath("vbn"))

rng(1)

%% INPUTS
t0 = 10;
tend = 16;
nt = (tend-t0)*15; % time vector length
nsp = 9; % propagator state
nsf = 6; % filter state dimension
nzf = 3; % filter measurement dimension

x0 = [-2; 2; 3; 1; -2; 2; 1; -1; 2]; % assume constant acceleration
x0 = [-2; 2; 3; 1; -2; 2; 0; 0; 0]; % assume constant velocity
sigma_pos0 = [4; 5; 3];
sigma_vel0 = [1; 2; 4];
sigma_z = [1; 1; 3];
sigma_dt = 0.01; % 1% error on timetag of the measurement

R_tuning = 1*sigma_z.^2;
P0 = diag([sigma_pos0.^2; sigma_vel0.^2]);

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
zTrue = xTrue(1:nzf, :);
zMeas = zTrue + sigma_z.*randn(size(zTrue));
dtMeas = dtTrue.*(1 + sigma_dt.*randn(size(tTrue)));
t0Meas = t0*(1 + sigma_dt.*randn(1));
tMeas = t0Meas + [0, cumsum(dtMeas(2:end))];

% Estimation
xEst = zeros(nsf, nt);
PEst = nan(nsf, nt);
xEst(1:3, 1) = xTrue(1:3,1) + sigma_pos0.*randn(3,1);
xEst(4:6, 1) = xTrue(4:6,1) + sigma_vel0.*randn(3,1);
PEst(1:6,1) = diag(P0);

% Initialize filter
flt = kf.xPosVel_zPos_dtVariable(xEst(:, 1), P0, tMeas(2) - tMeas(1));
flt.R_scaler = R_tuning;
flt.update_R;
for ix = 2:nt
    
    % Timestep extraction
    dt = tMeas(ix) - tMeas(ix-1);

    % Measurement 
    z = zMeas(:, ix);
    flagMeasNull = all(isnan(z));

    % flt.update_R;

    if ~flagMeasNull
        % predict and update
        flt.predict_and_update(z, dt);    
    else
        % predict only
        flt.predict_only(dt)
    end

    dtMeas(ix) = dt;
    xEst(:, ix) = flt.x_state;
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
    plot(tTrue, zMeas(ix,:), 'bo','LineWidth',2,'DisplayName',[txt_xyz{ix},'Meas'])
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

%---errors
figure('name','position_error'), 
txt_xyz = {'x','y','z'};
for ix = 1:3
    subplot(3,1,ix),grid on, hold on
    plot(tTrue, xTrue(ix,:) - zMeas(ix,:), 'b','LineWidth',2,'DisplayName',[txt_xyz{ix},'Meas Err'])
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
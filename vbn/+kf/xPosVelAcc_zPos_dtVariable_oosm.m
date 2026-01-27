classdef xPosVelAcc_zPos_dtVariable_oosm < kf
    %xPosVelAcc_zPos_dtVariableNegative Kalman Filter for tracking a 3D point
    % position, velocity and acceleration with position measurements and variable 
    % delayed timesteps. A buffer is included for retrodiction in case of 
    % out-of-sequence measurements (OOSM). This is an extended version where
    % all the measurements arrived after the OOSM are accounted for.
    
    properties
        id
        
        x_state    % [x; y; z; vx; vy; vz; ax; ay; az]
        time
        dt
        
        % Enhanced buffer stores measurements + gains for full reprocessing
        bufferSize = 5;  % Buffer size for OOSM history (modifiable)
        buffer = struct('x_pred', [], 'P_pred', [], 'F', [], 'Q', [], 'time', [], ...
                        'y', [], 'K', []);  % Circular buffer with measurements
        bufferPointer = 0;  % Current write index
    end
    
    methods
        function self = xPosVelAcc_zPos_dtVariable_oosm(x0, P0, dt0, bufferSize)
            self.id = 0;
            self.x_state = x0;  % 9D state: position, velocity, acceleration
            self.P = P0;
            self.time = 0;
            self.dt = dt0;
            % Measurement matrix assumes direct position measurement
            self.H = [1 0 0 0 0 0 0 0 0;
                      0 1 0 0 0 0 0 0 0;
                      0 0 1 0 0 0 0 0 0];
            self.R_scaler = eye(3);
            self.Q_scaler = eye(3);
            self = self.update_F();
            self = self.update_Q();
            self = self.update_R();
            
            % Initialize buffer with measurement history fields
            self.bufferSize = bufferSize;
            self.buffer = repmat(struct('x_pred', zeros(9,1), 'P_pred', eye(9), ...
                                        'F', eye(9), 'Q', zeros(9), 'time', 0, 'y', zeros(3,1), 'K', zeros(9,3)), ...
                                        [1 bufferSize]);
            self.bufferPointer = 0;
        end
        
        function self = update_F(self)
            % State transition matrix for constant acceleration model
            % Uses absolute value of dt for F structure (always forward kinematics)
            dt_abs = abs(self.dt);
            dt2 = dt_abs^2 / 2;
            self.F = eye(9);
            % Position to velocity
            self.F(1,4) = dt_abs;
            self.F(2,5) = dt_abs;
            self.F(3,6) = dt_abs;
            % Position to acceleration
            self.F(1,7) = dt2;
            self.F(2,8) = dt2;
            self.F(3,9) = dt2;
            % Velocity to acceleration
            self.F(4,7) = dt_abs;
            self.F(5,8) = dt_abs;
            self.F(6,9) = dt_abs;
        end
        
        function self = update_Q(self)
            % Process noise: simple model with acceleration noise
            % Elements for process noise covariance matrix (jerk noise model)
            % Uses absolute value of dt for process noise calculation
            dt_abs = abs(self.dt);
            Q11 = dt_abs^5 / 20;
            Q12 = dt_abs^4 / 8;
            Q13 = dt_abs^3 / 6;
            Q22 = dt_abs^3 / 3;
            Q23 = dt_abs^2 / 2;
            Q33 = dt_abs;
            Qsc = self.Q_scaler;
            if numel(Qsc) == 1
                self.Q = Qsc*[Q11*eye(3), Q12*eye(3), Q13*eye(3);
                              Q12*eye(3), Q22*eye(3), Q23*eye(3);
                              Q13*eye(3), Q23*eye(3), Q33*eye(3)];
            elseif numel(Qsc) == 3
                self.Q = [Qsc(1)*Q11*Qsc(1)*eye(3), Qsc(1)*Q12*Qsc(2)*eye(3), Qsc(1)*Q13*Qsc(3)*eye(3);
                          Qsc(2)*Q12*Qsc(1)*eye(3), Qsc(2)*Q22*Qsc(2)*eye(3), Qsc(2)*Q23*Qsc(3)*eye(3);
                          Qsc(3)*Q13*Qsc(1)*eye(3), Qsc(3)*Q23*Qsc(2)*eye(3), Qsc(3)*Q33*Qsc(3)*eye(3)];
            elseif numel(Qsc) == 9
                self.Q = [Qsc(1)*Q11*eye(3), Qsc(4)*Q12*eye(3), Qsc(7)*Q13*eye(3);
                          Qsc(2)*Q12*eye(3), Qsc(5)*Q22*eye(3), Qsc(8)*Q23*eye(3);
                          Qsc(3)*Q13*eye(3), Qsc(6)*Q23*eye(3), Qsc(9)*Q33*eye(3)];
            else
                error('Q_scaler should be size 1, 3, 3x3 or 9')
            end
        end
        
        function self = update_R(self)
            % Measurement noise scaler
            Rsc = self.R_scaler;
            if numel(Rsc) == 1
                self.R = Rsc * eye(3); 
            elseif numel(Rsc) == 3
                self.R = diag(Rsc);
            elseif numel(Rsc) == 9
                self.R = reshape(Rsc, 3, 3);
            else
                error('R_scaler should be size 1, 3, 3x3 or 9')
            end
        end
        
        function self = store_history(self, x_pred, P_pred, F, Q, time, y, K)
            % Store predicted state, covariance, measurement innovation and gain
            % Enables full reprocessing during OOSM repropagation
            self.bufferPointer = mod(self.bufferPointer, self.bufferSize) + 1;
            self.buffer(self.bufferPointer) = struct('x_pred', x_pred, 'P_pred', P_pred, ...
                'F', F, 'Q', Q, 'time', time, 'y', y, 'K', K);
        end
        
        function [lag_steps, t_found] = find_lag_by_time(self, time_target)
            % Find lag (number of steps) by matching target time in history buffer
            % Returns lag_steps and found time; lag_steps=0 if not found
            lag_steps = 0;
            t_found = nan;
            tol = 1e-9;
            
            for i = 1:self.bufferSize
                ptr = mod(self.bufferPointer - i, self.bufferSize);
                if ptr == 0; ptr = self.bufferSize; end
                h = self.buffer(ptr);
                
                if abs(h.time - time_target) < tol
                    lag_steps = i;
                    t_found = h.time;
                    return;
                end
            end
            
            % If exact match not found, find closest
            min_diff = inf;
            for i = 1:self.bufferSize
                ptr = mod(self.bufferPointer - i, self.bufferSize);
                if ptr == 0; ptr = self.bufferSize; end
                h = self.buffer(ptr);
                
                diff = abs(h.time - time_target);
                if diff < min_diff
                    min_diff = diff;
                    lag_steps = i;
                    t_found = h.time;
                end
            end
        end
                
        function self = predict_and_update(self, x_meas, dt)
            % Predict and update with automatic OOSM handling
            % If dt > 0: standard forward predict-update, store in history
            % If dt < 0: treat as OOSM, retrodict by time offset, apply measurement
            % Handles non-uniform timesteps via absolute time tracking
            
            if dt > 0
                % Forward filtering: standard predict-update
                self.dt = dt;
                self = self.update_F();
                self = self.update_Q();
                
                % Predict
                x_pred = self.F * self.x_state;
                self.P = self.F * self.P * self.F' + self.Q;
                self.time = self.time + dt;
                
                % Update
                S = self.H * self.P * self.H' + self.R;
                K = self.P * self.H' / S;
                y = x_meas - self.H * x_pred;
                self.x_state = x_pred + K * y;
                self.P = (eye(9) - K * self.H) * self.P;
                
                % Store predicted state + measurement history for retrodiction
                self = self.store_history(x_pred, self.P, self.F, self.Q, self.time, y, K);
                
            elseif dt < 0
                % Backward filtering: OOSM handling via retrodiction
                % Compute target time for this OOSM measurement
                t_meas = self.time + dt;
                
                % Find lag index by matching absolute time
                [lag_steps, t_found] = self.find_lag_by_time(t_meas);
                
                if lag_steps == 0
                    warning('OOSM measurement time %.6f not in buffer; skipped', t_meas);
                    return;
                end
                
                if lag_steps > self.bufferSize
                    warning('OOSM lag %d exceeds buffer size %d; clamping', lag_steps, self.bufferSize);
                    lag_steps = self.bufferSize;
                end
                
                % Apply retro-correction with full measurement history
                self = self.retro_update(x_meas, lag_steps);
            else
                % Update only (dt == 0)
                self = self.update_only(x_meas);
            end
        end
        
        function self = predict_only(self, dt)
            % Predict only with automatic OOSM detection
            % If dt > 0: standard forward predict, store in history
            % If dt < 0: retrodict only (no measurement update)
            % Handles non-uniform timesteps via absolute time tracking
            
            if dt > 0
                % Forward prediction: standard
                self.dt = dt;
                self = self.update_F();
                self = self.update_Q();
                
                % Predict
                x_pred = self.F * self.x_state;
                self.P = self.F * self.P * self.F' + self.Q;
                self.time = self.time + dt;
                
                % Store for potential retrodiction (no measurement on predict-only)
                self = self.store_history(x_pred, self.P, self.F, self.Q, self.time, zeros(3,1), zeros(9,3));
                
                % Update state
                self.x_state = x_pred;
                
            elseif dt < 0
                % Backward prediction: retrodict only (no measurement update)
                time_target = self.time + dt;
                
                % Find lag by matching absolute time
                [lag_steps, t_found] = self.find_lag_by_time(time_target);
                
                if lag_steps == 0
                    warning('Target time %.6f not in buffer; predict_only retrodict skipped', time_target);
                    return;
                end
                
                if lag_steps > self.bufferSize
                    warning('Retrodict lag %d exceeds buffer size %d; clamping', lag_steps, self.bufferSize);
                    lag_steps = self.bufferSize;
                end
                
                % Retrodict without measurement update
                [self.x_state, self.P] = self.retrodict(lag_steps);
            else
                % Do nothing (dt == 0)
            end
        end
        
        function self = update_only(self, x_meas)
            % Update measurement without prediction step
            x_pred = self.x_state;
            
            % Update
            S = self.H * self.P * self.H' + self.R;
            K = self.P * self.H' / S;
            y = x_meas - self.H * x_pred;
            self.x_state = x_pred + K * y;
            self.P = (eye(9) - K * self.H) * self.P;
        end
        
        function self = retro_update(self, x_oosm, lag_steps)
            % Apply OOSM correction via retrodiction + full measurement reprocessing
            % Retrodict to measurement time, update, then repropagate with original measurements
            
            [x_retro, P_retro] = self.retrodict(lag_steps);
            
            % Measurement update at retrodict time
            S = self.H * P_retro * self.H' + self.R;
            K = P_retro * self.H' / S;
            y_oosm = x_oosm - self.H * x_retro;
            x_corr = x_retro + K * y_oosm;
            P_corr = (eye(9) - K * self.H) * P_retro;
            
            % Propagate correction forward with ORIGINAL measurements reapplied
            for i = 1:lag_steps
                ptr = mod(self.bufferPointer - lag_steps + i, self.bufferSize);
                if ptr == 0; ptr = self.bufferSize; end
                
                h = self.buffer(ptr);
                
                % Forward kinematics
                x_corr = h.F * x_corr;
                P_corr = h.F * P_corr * h.F' + h.Q;
                
                % RE-APPLY original measurement if it exists
                if ~(all(h.y == 0) && all(all(h.K == 0)))  % Check if measurement was stored
                    % Recalculate gain with updated covariance
                    S_reuse = self.H * P_corr * self.H' + self.R;
                    K_reuse = P_corr * self.H' / S_reuse;
                    % Apply stored innovation
                    x_corr = x_corr + K_reuse * h.y;
                    P_corr = (eye(9) - K_reuse * self.H) * P_corr;
                end
            end
            
            % Replace current state with fully corrected state
            self.x_state = x_corr;
            self.P = P_corr;
        end
        
        function [x_retro, P_retro] = retrodict(self, lag_steps)
            % Retrodict state and covariance backward using stored forward history
            % lag_steps > 0: number of steps to go back
            
            if lag_steps <= 0
                x_retro = self.x_state;
                P_retro = self.P;
                return;
            end
            
            if lag_steps > self.bufferSize
                warning('Requested lag %d exceeds buffer size %d', lag_steps, self.bufferSize);
                lag_steps = self.bufferSize;
            end
            
            x_retro = self.x_state;
            P_retro = self.P;
            
            % Walk backward using stored forward F and Q
            for i = 1:lag_steps
                ptr = mod(self.bufferPointer - i, self.bufferSize);
                if ptr == 0; ptr = self.bufferSize; end
                
                h = self.buffer(ptr);
                F_inv = h.F;
                F_inv(1,4) = -F_inv(1,4);
                F_inv(2,5) = -F_inv(2,5);
                F_inv(3,6) = -F_inv(3,6);
                F_inv(4,7) = -F_inv(4,7);
                F_inv(5,8) = -F_inv(5,8);
                F_inv(6,9) = -F_inv(6,9);
                Q_adj = 0.9 * h.Q;  % Empirical factor for info removal
                
                x_retro = F_inv * x_retro;
                P_retro = F_inv * P_retro * F_inv' - Q_adj;
                
                % Ensure P_retro remains positive semi-definite
                P_retro = (P_retro + P_retro') / 2;
            end
        end
       
        function [x_state, P] = propagate(self, dt)
            % Propagate the state and covariance without setting it as a property
            % Uses forward kinematics only (dt should be positive)
            
            self.dt = dt;
            self = self.update_F();
            self = self.update_Q();
            x_state = self.F * self.x_state;
            P = self.F * self.P * self.F' + self.Q;
        end
    end
end

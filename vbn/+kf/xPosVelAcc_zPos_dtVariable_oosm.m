classdef xPosVelAcc_zPos_dtVariable_oosm < kf
    %xPosVelAcc_zPos_dtVariableNegative Kalman Filter for tracking a 3D point
    % position, velocity and acceleration with position measurements and variable 
    % delayed timesteps. Implements OOSM Reprocessing (Store State & Re-filter).
    
    properties
        id
        
        x_state    % [x; y; z; vx; vy; vz; ax; ay; az]
        time
        dt
        
        bufferSize        % Fixed size buffer
        buffer            % Buffer struct: x_state, P, time, x_meas
        bufferPointer
    end
    
    methods
        function self = xPosVelAcc_zPos_dtVariable_oosm(x0, P0, dt0)
            self.id = 0;
            self.x_state = x0; 
            self.P = P0;
            self.time = 0;
            self.dt = dt0;
            
            % Measurement matrix (Position only)
            self.H = [1 0 0 0 0 0 0 0 0;
                      0 1 0 0 0 0 0 0 0;
                      0 0 1 0 0 0 0 0 0];
                      
            self.R_scaler = eye(3);
            self.Q_scaler = eye(3);
            self = self.update_F();
            self = self.update_Q();
            self = self.update_R();
            
            % Initialize buffer structure
            self.bufferSize = 64;
            self.buffer = repmat(struct('x_state', zeros(9,1), 'P', eye(9), ...
                                 'time', 0, 'x_meas', zeros(3,1)), [1 64]);
            self.bufferPointer = 0;
            
            % Store initial state as the first history point
            self = self.store_history(x0, P0, 0, zeros(3,1));
        end
        
        function self = update_F(self)
            % State transition matrix for constant acceleration model
            % When dt is negative, F is backward dynamics
            dt_temp = self.dt;
            dt2 = dt_temp^2 / 2;
            self.F = eye(9);
            % Pos -> Vel
            self.F(1,4) = dt_temp; self.F(2,5) = dt_temp; self.F(3,6) = dt_temp;
            % Pos -> Acc
            self.F(1,7) = dt2;    self.F(2,8) = dt2;    self.F(3,9) = dt2;
            % Vel -> Acc
            self.F(4,7) = dt_temp; self.F(5,8) = dt_temp; self.F(6,9) = dt_temp;
        end
        
        function self = update_Q(self)
            % Process noise: simple model with acceleration noise
            % Elements for process noise covariance matrix (jerk noise model)
            % Uses absolute value of dt for process noise calculation
            dt_abs = abs(self.dt);
            Q11 = dt_abs^5 / 20; Q12 = dt_abs^4 / 8; Q13 = dt_abs^3 / 6;
            Q22 = dt_abs^3 / 3;  Q23 = dt_abs^2 / 2; Q33 = dt_abs;
            
            Qsc = self.Q_scaler;
            % % Construct Q based on scaler size
            % if numel(Qsc) == 1
            %     self.Q = Qsc * [Q11*eye(3), Q12*eye(3), Q13*eye(3);
            %                     Q12*eye(3), Q22*eye(3), Q23*eye(3);
            %                     Q13*eye(3), Q23*eye(3), Q33*eye(3)];
            % elseif numel(Qsc) == 3
            %     self.Q = [Qsc(1)*Q11*Qsc(1)*eye(3), Qsc(1)*Q12*Qsc(2)*eye(3), Qsc(1)*Q13*Qsc(3)*eye(3);
            %                Qsc(2)*Q12*Qsc(1)*eye(3), Qsc(2)*Q22*Qsc(2)*eye(3), Qsc(2)*Q23*Qsc(3)*eye(3);
            %                Qsc(3)*Q13*Qsc(1)*eye(3), Qsc(3)*Q23*Qsc(2)*eye(3), Qsc(3)*Q33*Qsc(3)*eye(3)];
            % else
                self.Q = [Qsc(1)*Q11*eye(3), Qsc(4)*Q12*eye(3), Qsc(7)*Q13*eye(3);
                          Qsc(2)*Q12*eye(3), Qsc(5)*Q22*eye(3), Qsc(8)*Q23*eye(3);
                          Qsc(3)*Q13*eye(3), Qsc(6)*Q23*eye(3), Qsc(9)*Q33*eye(3)];
            % end
        end
        
        function self = update_R(self)
            Rsc = self.R_scaler;
            % if numel(Rsc) == 1;     self.R = Rsc * eye(3); 
            % elseif numel(Rsc) == 3; self.R = diag(Rsc);
            % else                   
                self.R = reshape(Rsc, 3, 3);
            % end
        end
        
        function self = store_history(self, x_state, P, time, x_meas)
            % Store Posterior State and Raw Measurement for reprocessing
            self.bufferPointer = mod(self.bufferPointer, self.bufferSize) + 1;
            self.buffer(self.bufferPointer) = struct('x_state', x_state, ...
                'P', P, 'time', time, 'x_meas', x_meas);
        end

        function [idx, found] = find_previous_state_index(self, t_target)
            % Find the most recent buffer state index where time <= t_target
            
            % 1. Extract times from buffer structure array
            all_times = [self.buffer.time];
            
            % 2. Calculate difference: (Stored Time - Target Time)
            % Negative values are in the past. Positive values are in the future.
            diffs = all_times - t_target;
            
            % 3. Mask valid candidates: 
            % - Must be in the past (diffs <= tolerance)
            % - Must be a valid initialized slot (time >= 0)
            valid_mask = (diffs <= 1e-9) & (all_times >= 0);
            
            if ~any(valid_mask)
                idx = -1;
                found = false;
                return;
            end
            
            % 4. Find the "Maximum" of the Negative differences
            % This corresponds to the time closest to t_target from below.
            
            % Set invalid entries to -infinity so max() ignores them
            search_diffs = diffs;
            search_diffs(~valid_mask) = -inf;
            
            [~, idx] = max(search_diffs);
            found = true;
        end
        
        function [indices, nind] = get_sorted_indices_after(self, t_threshold)
            % Helper to get buffer indices strictly after t_threshold, sorted by time
            
            % Preallocate
            indices = zeros(1, self.bufferSize);
            nind = 0;

            % Find indices where time is strictly greater than threshold
            for k = 1:self.bufferSize
                tk = self.buffer(k).time;
                if tk > t_threshold + 1e-9
                    nind = nind + 1;
                    indices(nind) = k;
                end
            end
            if nind == 0
                return;
            end
            
            % Sort the found indices based on their corresponding times
            times_valid = zeros(1, nind);
            for i = 1:nind
                times_valid(i) = self.buffer(indices(i)).time;
            end
            [~, sort_order] = sort(times_valid);
            indices(1:nind) = indices(sort_order);
        end
                
        function self = predict_and_update(self, x_meas, dt)
            % Handles both Forward (dt>0) and OOSM (dt<0)
            
            if dt > 0
                % --- STANDARD FORWARD FILTERING ---
                self.dt = dt;
                self = self.update_F();
                self = self.update_Q();
                
                % Predict
                x_pred = self.F * self.x_state;
                P_pred = self.F * self.P * self.F' + self.Q;
                
                % Update
                S = self.H * P_pred * self.H' + self.R;
                K = P_pred * self.H' / S;
                y = x_meas - self.H * x_pred;
                self.x_state = x_pred + K * y;
                self.P = (eye(9) - K * self.H) * P_pred;
                
                % Store the state for future reprocessing
                self.time = self.time + dt;
                self = self.store_history(self.x_state, self.P, self.time, x_meas);
            
            elseif dt == 0
                % --- UPDATE ONLY ---
                self = update_only(self, x_meas);

            elseif dt < 0
                % --- OOSM REPROCESSING ---
                
                t_oosm = self.time + dt;
                
                % 1. Find the base state in buffer (closest time <= t_oosm)
                [base_idx, found] = self.find_previous_state_index(t_oosm);
                
                if ~found
                    warning('OOSM too old or buffer empty. Skipped.');
                    return;
                end
                
                % 2. Reprocess sequence starting from base state
                self = self.reprocess_sequence(base_idx, t_oosm, x_meas);
            end

        end
        
        function self = reprocess_sequence(self, start_idx, t_oosm, x_oosm)
            % Load Base State (last valid state before OOSM)
            h = self.buffer(start_idx);
            curr_x = h.x_state;
            curr_P = h.P;
            curr_t = h.time;
            
            % 1. Propagate from Base State -> OOSM Time
            dt_gap = t_oosm - curr_t;
            if dt_gap > 1e-9
                self.dt = dt_gap;
                self = self.update_F();
                self = self.update_Q();
                curr_x = self.F * curr_x;
                curr_P = self.F * curr_P * self.F' + self.Q;
            end
            
            % 2. Update with OOSM measurement
            S = self.H * curr_P * self.H' + self.R;
            K = curr_P * self.H' / S;
            y = x_oosm - self.H * curr_x;
            curr_x = curr_x + K * y;
            curr_P = (eye(9) - K * self.H) * curr_P;
            curr_t = t_oosm;
            
            % 3. Re-propagate through history to Present
            % Get all subsequent measurements from buffer
            [indices, nind] = self.get_sorted_indices_after(t_oosm);
            
            for k = 1:nind
                idx = indices(k);
                h_next = self.buffer(idx);
                
                % Predict to next stored time
                dt_step = h_next.time - curr_t;
                if dt_step > 1e-9
                    self.dt = dt_step;
                    self = self.update_F();
                    self = self.update_Q();
                    curr_x = self.F * curr_x;
                    curr_P = self.F * curr_P * self.F' + self.Q;
                end
                
                % Re-Update with stored measurements
                S = self.H * curr_P * self.H' + self.R;
                K = curr_P * self.H' / S;
                y = h_next.x_meas - self.H * curr_x; 
                curr_x = curr_x + K * y;
                curr_P = (eye(9) - K * self.H) * curr_P;
                
                curr_t = h_next.time;
                
                % Re-write buffer with improved state
                self.buffer(idx).x_state = curr_x;
                self.buffer(idx).P = curr_P;
            end
            
            % Restore final state to object
            self.x_state = curr_x;
            self.P = curr_P;
        end
        
        function self = predict_only(self, dt)
            % Simplified predict_only.
            
            if dt > 0
                self.dt = dt;
                self = self.update_F();
                self = self.update_Q();
                
                x_pred = self.F * self.x_state;
                self.P = self.F * self.P * self.F' + self.Q;
                self.x_state = x_pred;
                
                self.time = self.time + dt;
            else
                warning('Negative or zero delta. Skipped.');
            end
        end
        
        function self = update_only(self, x_meas)
            % Update step only. When you receive multiple measures at the
            % same time, after the first update you don't propagate

            x_pred = self.x_state;
            S = self.H * self.P * self.H' + self.R;
            K = self.P * self.H' / S;
            y = x_meas - self.H * x_pred;
            self.x_state = x_pred + K * y;
            self.P = (eye(9) - K * self.H) * self.P;
        end
        
        function [x_state, P] = propagate(self, dt, flag_zero_acceleration)
            % Propagate the state and covariance without setting it as a property
            % Optional arg: flag_zero_acceleration (default false)
            % If true, temporarily sets acceleration to 0 for a Uniform Velocity propagation.
            
            if nargin < 3
                flag_zero_acceleration = false;
            end
            
            self.dt = dt;
            self = self.update_F();
            self = self.update_Q();
            
            % Create local copy of state
            x_local = self.x_state;
            
            if flag_zero_acceleration
                % Force acceleration terms (indices 7, 8, 9) to zero
                x_local(7:9) = 0;
            end
            
            % Standard propagation applies CV model automatically since acc is 0
            x_state = self.F * x_local;
            P = self.F * self.P * self.F' + self.Q;
        end
    end
end
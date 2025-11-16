classdef xPosVelAcc_zUVD_dtVariable < ekf
    %xPosVelAcc_zUVD_dtVariable EKF for tracking a 3D point
    % position, velocity and acceleration with image coordinates U, V and depth
    % measurements and with variable timestep.

    properties
        id

        x    % [x; y; z; vx; vy; vz; ax; ay; az]
        dt
        K          % intrinsic matrix
    end

    methods
        function self = xPosVelAcc_zUVD_dtVariable(x0, P0, dt0, K)

            self.id = 0;

            self.x = x0;  % [pos; vel; acc]
            self.P = P0;
            self.dt = dt0;
            self.K = K;
            
            self.R_scaler = 1.0;
            self.Q_scaler = 1.0;

            self = self.update_F();
            self = self.update_Q();
            self = self.update_R();
        end

        function self = update_F(self)
            dt2 = self.dt^2 / 2;
            self.F = eye(9);
            self.F(1,4) = self.dt;
            self.F(2,5) = self.dt;
            self.F(3,6) = self.dt;
            self.F(1,7) = dt2;
            self.F(2,8) = dt2;
            self.F(3,9) = dt2;
            self.F(4,7) = self.dt;
            self.F(5,8) = self.dt;
            self.F(6,9) = self.dt;
        end

        function self = update_Q(self)
            % Process noise: simple model with acceleration noise

            % Elements for process noise covariance matrix (jerk noise model)
            Q11 = self.dt^5 / 20;
            Q12 = self.dt^4 / 8;
            Q13 = self.dt^3 / 6;
            Q22 = self.dt^3 / 3;
            Q23 = self.dt^2 / 2;
            Q33 = self.dt;

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
            % Measurement noise scaler for U, V, D

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

        function h = measurement_function(self, x)
            % Nonlinear measurement model: [U; V; D]
            pos = x(1:3);
            x_pos = pos(1); y_pos = pos(2); z_pos = pos(3);

            U = self.K(1,1) * x_pos / z_pos + self.K(1,3);
            V = self.K(2,2) * y_pos / z_pos + self.K(2,3);
            D = norm(pos);

            h = [U; V; D];
        end

        function H = measurement_jacobian(self, x)
            % Jacobian of measurement model h(x) with respect to state x
            % Only first 3 components (position) are involved

            x_pos = x(1); y_pos = x(2); z_pos = x(3);
            r = norm(x(1:3));
            z2 = z_pos^2;

            H = zeros(3,9);
            % ∂U/∂x, ∂U/∂y, ∂U/∂z
            H(1,1) = self.K(1,1) / z_pos;
            H(1,2) = 0;
            H(1,3) = -self.K(1,1) * x_pos / z2;

            % ∂V/∂x, ∂V/∂y, ∂V/∂z
            H(2,1) = 0;
            H(2,2) = self.K(2,2) / z_pos;
            H(2,3) = -self.K(2,2) * y_pos / z2;

            % ∂D/∂x, ∂D/∂y, ∂D/∂z
            H(3,1:3) = [x_pos, y_pos, z_pos] / r;
        end

        function self = predict_and_update(self, z_meas, dt)
            self.dt = dt;
            self = self.update_F();
            self = self.update_Q();

            % Predict
            x_pred = self.F * self.x;
            P_pred = self.F * self.P * self.F' + self.Q;

            % Measurement prediction
            h_pred = self.measurement_function(x_pred);
            H = self.measurement_jacobian(x_pred);

            % Innovation
            y = z_meas - h_pred;
            S = H * P_pred * H' + self.R;
            Kgain = P_pred * H' / S;

            % Update
            self.x = x_pred + Kgain * y;
            self.P = (eye(9) - Kgain * H) * P_pred;
        end

        function self = predict_only(self, dt)
            self.dt = dt;
            self = self.update_F();
            self = self.update_Q();

            self.x = self.F * self.x;
            self.P = self.F * self.P * self.F' + self.Q;
        end
    end
end

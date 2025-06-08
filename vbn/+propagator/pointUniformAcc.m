classdef pointUniformAcc < propagator
    % pointUniformAcc: 3D point propagator with uniform acceleration
    % State vector x = [position; velocity; acceleration], 9x1

    properties
        x   % 9x1 state vector: [position; velocity; acceleration]
    end

    methods
        function obj = pointUniformAcc(position, velocity, acceleration)
            % Constructor
            if nargin == 3
                obj.x = [position(:); velocity(:); acceleration(:)];
            else
                obj.x = zeros(9, 1);
            end
        end

        function propagate(obj, dt)
            % PROPAGATE: advance state with constant acceleration
            if ~isscalar(dt)
                error('Time step dt must be a scalar.');
            end

            pos = obj.x(1:3);
            vel = obj.x(4:6);
            acc = obj.x(7:9);

            pos_new = pos + vel * dt + 0.5 * acc * dt^2;
            vel_new = vel + acc * dt;

            obj.x = [pos_new; vel_new; acc];
        end

        function p = getPosition(obj)
            p = obj.x(1:3);
        end

        function v = getVelocity(obj)
            v = obj.x(4:6);
        end

        function a = getAcceleration(obj)
            a = obj.x(7:9);
        end

        function set.x(obj, value)
            % Validate state vector
            if ~isvector(value) || numel(value) ~= 9
                error('State vector x must be a 9-element vector: [position; velocity; acceleration]');
            end
            obj.x = value(:);
        end
    end
end

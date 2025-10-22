function nlmpc_ddrive_circle_demo()
% Differential-drive NL-MPC (no ACADO) — Blue robot tracking a blue circle.
% Robust printing (works across MATLAB versions), full-horizon refs,
% auto-feasible circle speed, cruise→MPC handover, red trail + live time.

%% ---------------- User knobs ----------------
Ts     = 0.05;          % controller sample time [s]
Tfinal = 40;            % shorter run for quick testing
Hp     = 20;            % prediction horizon (short = faster)
Hc     = 10;            % control horizon (short = faster)

% Robot geometry / scalings
L=0.50; alphaL=1.0; alphaR=1.0; xicr=0.0;

% Limits
vmin=-1.0; vmax=1.0;    % wheel-speed limits [m/s]
amin=-3.0; amax=3.0;    % wheel accel limits [m/s^2]
damin=-20; damax=20;    % accel rate limits [m/s^3]

% ---- Circle (auto-feasible with given vmax) ----
R_des   = 2.0;
omega_d = 0.50;
omega_max = vmax / (R_des + 0.5*L);
if omega_d > omega_max
    warning('omega=%.3f > feasible %.3f for vmax=%.2f; reducing omega.', ...
            omega_d, omega_max, vmax);
end
omega = min(omega_d, 0.95*omega_max);   % small margin
R     = R_des;

% Steady wheel speeds for that circle
v_lin   = R*omega;
vL_ref0 = v_lin - 0.5*L*omega;
vR_ref0 = v_lin + 0.5*L*omega;

% Animation pacing (visual only)
pause_time = 0.00;      % set 0.01 to slow the display

% Phase timings (see motion immediately)
t_ramp   = 1.0;   % s (spin-up)
t_cruise = 4.0;   % s (open-loop constant speed you can SEE)
rampSteps   = round(t_ramp/Ts);
cruiseSteps = round(t_cruise/Ts);

%% ---------------- Build references ----------------
t    = 0:Ts:Tfinal;
xRef = R*cos(omega*t);
yRef = R*sin(omega*t);
thRef= wrapToPiLocal(omega*t + pi/2);

%% ---------------- NL MPC (nx=6, ny=3, nu=2) ----------------
nx=6; ny=3; nu=2;
nlobj = nlmpc(nx,ny,nu);
nlobj.Ts=Ts; nlobj.PredictionHorizon=Hp; nlobj.ControlHorizon=Hc;

% State update: x=[x;y;th; w; vL; vR], u=[aL; aR]
nlobj.Model.StateFcn  = @(x,u) ddrive_step_midpoint(x,u,Ts,L,alphaL,alphaR,xicr);
nlobj.Model.OutputFcn = @(x,u) x(1:3);      % track pose only

% MV limits
nlobj.MV(1).Min=amin; nlobj.MV(1).Max=amax;
nlobj.MV(2).Min=amin; nlobj.MV(2).Max=amax;
nlobj.MV(1).RateMin=damin; nlobj.MV(1).RateMax=damax;
nlobj.MV(2).RateMin=damin; nlobj.MV(2).RateMax=damax;

% Wheel-speed bounds as hard constraints
nlobj.Optimization.CustomIneqConFcn = @(X,U,e,data) v_bounds_ineq(X, vmin, vmax);

% Weights
nlobj.Weights.OutputVariables          = [350 350 7];
nlobj.Weights.ManipulatedVariables     = [0.01 0.01];
nlobj.Weights.ManipulatedVariablesRate = [0.04 0.04];

try
  nlobj.Optimization.SolverOptions = optimoptions('fmincon','Algorithm','sqp', ...
     'MaxIterations',40,'ConstraintTolerance',1e-3,'OptimalityTolerance',1e-3, ...
     'StepTolerance',1e-6,'Display','none');
catch, end

% Start EXACTLY on the circle with correct steady wheel speeds
x0 = [ xRef(1); yRef(1); thRef(1); omega; vL_ref0; vR_ref0 ];
u0 = [0;0];
validateFcns(nlobj, x0, u0);
opt = nlmpcmoveopt;

%% ---------------- Closed-loop sim & animation ----------------
N=numel(t)-1; X=zeros(nx,N+1); X(:,1)=x0; U=zeros(nu,N);

% Figure
figure('Color','w'); hold on; axis equal; grid on
plot(xRef,yRef,'b--','LineWidth',1.5);                             % reference circle
traj = plot(X(1,1), X(2,1), 'r-','LineWidth',2);                   % red trail
robot= patch('XData',[],'YData',[],'FaceColor',[0 0.4 1], ...
             'EdgeColor',[0 0.4 1],'FaceAlpha',0.45);
xlabel('x [m]'); ylabel('y [m]');
ttl = title(sprintf('NL MPC — t = %.1f s',0));
xlim(R*[-1.4,1.4]); ylim(R*[-1.4,1.4]);

fprintf('Start: omega=%.3f, vL0=%.3f, vR0=%.3f, Hp=%d, Hc=%d\n', omega, vL_ref0, vR_ref0, Hp, Hc);

for k=1:N
    % Full-horizon reference (look-ahead)
    kH = (k+1):(k+Hp);  kH(kH>numel(t)) = numel(t);
    YH = [ xRef(kH).' , yRef(kH).' , thRef(kH).' ];   % Hp x 3

    % Feed-forward toward steady circle speeds
    vL = X(5,k); vR = X(6,k);
    a_ff = [(vL_ref0 - vL)/Ts; (vR_ref0 - vR)/Ts];
    a_ff = max(min(a_ff,[amax;amax]),[amin;amin]);

    % Choose phase
    if k <= rampSteps
        uk = a_ff;                                    % RAMP
        phase = "RAMP ";
        solve_t = NaN; exit_str = "NA";
    elseif k <= rampSteps + cruiseSteps
        uk = [0; 0];                                  % CRUISE (constant speed)
        phase = "CRUISE";
        solve_t = NaN; exit_str = "NA";
    else
        phase = "MPC  ";
        tic;
        try
            [uk, info] = nlmpcmove(nlobj, X(:,k), a_ff, YH, [], opt);
        catch
            [uk, info] = nlmpcmove(nlobj, X(:,k), a_ff, YH.', [], opt);
        end
        solve_t = toc;

        % robust success check
        bad = any(~isfinite(uk)) || norm(uk) < 1e-7;
        if isstruct(info) && isfield(info,'ExitFlag'), bad = bad || info.ExitFlag<=0; end
        if bad, uk = a_ff; end

        % robust ExitFlag print (works even if info is an object)
        exit_str = exitflag_to_str(info);
        if mod(k,10)==0
            fprintf('k=%4d  phase=%s  solve=%.3fs  Exit=%s\n', k, phase, solve_t, exit_str);
        end
    end

    U(:,k)=uk;
    X(:,k+1)=ddrive_step_midpoint(X(:,k), uk, Ts, L, alphaL, alphaR, xicr);

    % Animate + live time
    set(traj,'XData',X(1,1:k+1),'YData',X(2,1:k+1));
    [vx,vy]=robotTriangle(X(1,k+1),X(2,k+1),X(3,k+1)); set(robot,'XData',vx,'YData',vy);
    set(ttl,'String',sprintf('NL MPC — t = %.1f s', k*Ts));
    drawnow; if pause_time>0, pause(pause_time); end

    % Per-step status (compact)
    fprintf('k=%4d  %s  a=[%5.2f %5.2f]  v=[%5.2f %5.2f]  pos=[%6.2f %6.2f]  th=%6.2f\n', ...
        k, phase, uk(1),uk(2), X(5,k+1),X(6,k+1), X(1,k+1),X(2,k+1), X(3,k+1));
end

% RMS after warm-up (ignore first 5 s)
posErr=hypot(X(1,1:end-1)-xRef(1:end-1), X(2,1:end-1)-yRef(1:end-1));
warm=round(5/Ts);
fprintf('RMS pos err (after warm-up)= %.3f m\n', sqrt(mean(posErr(warm+1:end).^2)));
end

%% ---- Helpers ----
function xk1=ddrive_step_midpoint(xk,uk,Ts,L,aL,aR,xicr)
% Midpoint integrator; enforce w=(vR-vL)/L at step end.
x=xk(1); y=xk(2); th=xk(3); vL=xk(5); vR=xk(6);
aLl=uk(1); aRr=uk(2);
vL1=vL+Ts*aLl; vR1=vR+Ts*aRr;
vLmid=0.5*(vL+vL1); vRmid=0.5*(vR+vR1); wmid=(vRmid-vLmid)/L;
vxmid=0.5*(aL*vLmid + aR*vRmid);
x1=x+Ts*(vxmid*cos(th)+xicr*sin(th)*wmid);
y1=y+Ts*(vxmid*sin(th)-xicr*cos(th)*wmid);
th1=wrapToPiLocal(th+Ts*wmid);
w1=(vR1-vL1)/L;
xk1=[x1; y1; th1; w1; vL1; vR1];
end

function ineq=v_bounds_ineq(X,vmin,vmax)
% Wheel-speed bounds along the horizon (<=0 format)
vL=X(:,5); vR=X(:,6);
ineq=[vL - vmax; vmin - vL; vR - vmax; vmin - vR];
end

function [vx,vy]=robotTriangle(xr,yr,th)
Lr=0.6; Wr=0.35;
B=[ Lr,0; -Lr/2,Wr/2; -Lr/2,-Wr/2 ]';
Rz=[cos(th),-sin(th); sin(th),cos(th)];
Wp=Rz*B+[xr;yr]; vx=Wp(1,:); vy=Wp(2,:);
end

function a=wrapToPiLocal(a), a=mod(a+pi,2*pi)-pi; end

function s = exitflag_to_str(info)
% Robustly turn ExitFlag into string across MATLAB versions.
s = "NA";
try
    if isstruct(info) && isfield(info,'ExitFlag')
        s = string(info.ExitFlag);
    elseif ~isstruct(info)
        % some versions return an object (attempt property access)
        if isprop(info,'ExitFlag')
            s = string(info.ExitFlag);
        end
    end
catch
    % leave as "NA"
end
end

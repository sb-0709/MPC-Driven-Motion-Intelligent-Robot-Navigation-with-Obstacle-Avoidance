function nlmpc_spline_follow_demo(mode, track_id)
% NL-MPC spline path following (fast + accurate).
% - Frenet-frame custom cost (cross-track + heading)
% - Forward-only projection (no jumping at bends/crossings)
% - Deterministic arc-length progress with capped preview step
% - Typo fix in integrator (vR1)

if nargin<1 || isempty(mode), mode='realtime'; end
if nargin<2 || isempty(track_id), track_id=1; end
RUN_TURBO = strcmpi(mode,'turbo');

%% ---------- User knobs (tuned for lower RMS) ----------
Ts       = 0.04;       % controller sample time
Tfinal   = 60;         % hard cap on sim time
Hp       = 24;         % prediction horizon (steps)
Hc       = 10;         % control horizon (steps)
drawEvery  = 10;       % plot every N steps (left panel)
solveEvery = 1;        % solve MPC every step (best tracking)
decimateAccum = 3;     % downsample factor for "Accumulated" panel

% Robot / limits
L=0.50; alphaL=1.0; alphaR=1.0; xicr=0.0;
vmin=-1.0; vmax=1.0;                 % wheel speeds [m/s]
amin=-3.5; amax=3.5;                 % wheel accels [m/s^2]
damin=-50;  damax=50;                % accel rate limits [m/s^3]

% Desired forward speed (curvature cap applied inside loop)
v_nom    = 0.75;                     % conservative nominal
v_margin = 0.95;

% Spline sampling density
Ngrid = 1500;

stop_on_finish = true;

%% ---------- Build spline path (unwrapped tangent) ----------
[W, closed] = default_waypoints(track_id);
path = buildSplinePath(W, closed, Ngrid);

%% ---------- NL-MPC set-up ----------
nx=6; ny=3; nu=2;
nlobj = nlmpc(nx,ny,nu);
nlobj.Ts = Ts; nlobj.PredictionHorizon = Hp; nlobj.ControlHorizon = Hc;

% State update: x=[x;y;theta; w; vL; vR], u=[aL; aR]
nlobj.Model.StateFcn  = @(x,u) ddrive_step_midpoint(x,u,Ts,L,alphaL,alphaR,xicr);
nlobj.Model.OutputFcn = @(x,u) x(1:3);   % defined but cost is custom

% MV limits
nlobj.MV(1).Min=amin; nlobj.MV(1).Max=amax;
nlobj.MV(2).Min=amin; nlobj.MV(2).Max=amax;
nlobj.MV(1).RateMin=damin; nlobj.MV(1).RateMax=damax;
nlobj.MV(2).RateMin=damin; nlobj.MV(2).RateMax=damax;

% Wheel-speed bounds as hard constraints
nlobj.Optimization.CustomIneqConFcn = @(X,U,e,data) v_bounds_ineq(X, vmin, vmax);

% Use ONLY custom cost (avoid built-in LSQ)
nlobj.Weights.OutputVariables          = [0 0 0];
nlobj.Weights.ManipulatedVariables     = [0 0];
nlobj.Weights.ManipulatedVariablesRate = [0 0];

try
  nlobj.Optimization.SolverOptions = optimoptions('fmincon','Algorithm','sqp', ...
     'MaxIterations',35,'ConstraintTolerance',1e-3,'OptimalityTolerance',2e-3, ...
     'StepTolerance',1e-6,'Display','none');
catch, end

%% ---------- Start ON the path (consistent steady speeds) ----------
s0   = 0;
[xr,yr,thr,kapr] = samplePath(path, s0);
vff  = min(v_nom, vmax/(1+0.5*L*abs(kapr)));
wff  = kapr*vff;
vL0  = vff - 0.5*L*wff;  vR0 = vff + 0.5*L*wff;

x0 = [xr; yr; thr; wff; vL0; vR0];
u0 = [0;0];
validateFcns(nlobj, x0, u0);
opt = nlmpcmoveopt;

%% ---------- Sim buffers ----------
t  = 0:Ts:Tfinal; N = numel(t)-1;
X  = zeros(nx,N+1); X(:,1)=x0;
U  = zeros(nu,N);

% reference arc-length progress
snow = s0;
ds_max = 0.70 * v_nom * Ts;        % capped preview step

% forward-only nearest search bookkeeping
idx_last = 1;
ds_step  = mean(diff(path.s));
lookAhead = max(10, ceil((1.5*v_nom*Ts)/max(ds_step,1e-6)));  % indices window

%% ---------- Figures ----------
if ~RUN_TURBO
    tl = tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
    % realtime (left)
    ax1 = nexttile(tl,1); hold(ax1,'on'); axis(ax1,'equal'); grid(ax1,'on');
    plot(ax1,path.x, path.y,'b--','LineWidth',1.3);
    traj1 = plot(ax1,X(1,1), X(2,1), 'r-','LineWidth',2);
    robot = patch(ax1,'XData',[],'YData',[], ...
                  'FaceColor',[0 0.4 1],'EdgeColor',[0 0.4 1],'FaceAlpha',0.45);
    title1 = title(ax1,sprintf('Realtime — t=%.1fs',0));
    xlabel(ax1,'x [m]'); ylabel(ax1,'y [m]');
    m=1.0; xlim(ax1,[min(path.x)-m, max(path.x)+m]); ylim(ax1,[min(path.y)-m, max(path.y)+m]);
    % accumulated (right)
    ax2 = nexttile(tl,2); hold(ax2,'on'); axis(ax2,'equal'); grid(ax2,'on');
    plot(ax2,path.x, path.y,'b--','LineWidth',1.3);
    traj2 = plot(ax2,X(1,1), X(2,1), 'r-','LineWidth',2);
    title(ax2,'Accumulated'); xlabel(ax2,'x [m]'); ylabel(ax2,'y [m]');
    xlim(ax2,[min(path.x)-m, max(path.x)+m]); ylim(ax2,[min(path.y)-m, max(path.y)+m]);
end

%% ---------- Main loop ----------
uk_prev = u0;

for k=1:N
    % ----- Forward-only nearest projection -----
    idx = nearestIndexForward(path.x, path.y, X(1,k), X(2,k), idx_last, lookAhead);
    idx_last = idx;
    s_proj = path.s(idx);

    % ----- Deterministic progress + clamp to projection -----
    snow = snow + min(v_nom*Ts, ds_max);
    snow = max(snow, s_proj);
    snow = min(snow, path.s_end);

    % ----- Build horizon from snow forward (capped step) -----
    sH = zeros(Hp,1); xH=sH; yH=sH; thH=sH; kapH=sH; vcapH=sH;
    sH(1) = snow;
    for i=1:Hp
        [xH(i),yH(i),thH(i),kapH(i)] = samplePath(path, sH(i));
        vcapH(i) = min(v_nom, vmax/(1+0.5*L*abs(kapH(i))));
        if i<Hp
            ds = min(v_margin*vcapH(i)*Ts, ds_max);
            sH(i+1) = min(path.s_end, sH(i) + ds);
        end
    end

    % unwrap heading targets around current heading
    thH = unwrapAround(thH, X(3,k));
    YH  = [xH, yH, thH];    % Hp x 3 (used by custom cost)

    % curvature-based feedforward at start of horizon
    vff  = min(v_nom, vmax/(1+0.5*L*abs(kapH(1))));
    wff  = kapH(1)*vff;
    vLff = vff - 0.5*L*wff;  vRff = vff + 0.5*L*wff;
    a_ff = [ (vLff - X(5,k))/Ts ; (vRff - X(6,k))/Ts ];
    a_ff = max(min(a_ff,[amax;amax]),[amin;amin]);

    % set Frenet custom cost for THIS horizon (ref passed via closure)
    nlobj.Optimization.CustomCostFcn = @(Xpred,Upred,e,data) frenetCostLocal(Xpred,Upred,e,data,YH);

    % solve MPC
    if mod(k-1, solveEvery)==0
        [uk, ~] = nlmpcmove(nlobj, X(:,k), a_ff, [], [], opt);  % ref handled in custom cost
        if any(~isfinite(uk)) || norm(uk)<1e-8, uk = a_ff; end
        uk_prev = uk;
    else
        uk = uk_prev;
    end

    % integrate one step
    U(:,k)=uk;
    X(:,k+1)=ddrive_step_midpoint(X(:,k), uk, Ts, L, alphaL, alphaR, xicr);

    % draw
    if ~RUN_TURBO && (mod(k,drawEvery)==0 || k==N)
        set(traj1,'XData',X(1,1:k+1),'YData',X(2,1:k+1));
        [vx,vy]=robotTriangle(X(1,k+1),X(2,k+1),X(3,k+1)); set(robot,'XData',vx,'YData',vy);
        set(title1,'String',sprintf('Realtime — t=%.1fs', k*Ts));
        d=decimateAccum; set(traj2,'XData',X(1,1:d:k+1),'YData',X(2,1:d:k+1));
        drawnow limitrate nocallbacks;
    end

    if stop_on_finish && snow >= path.s_end - 1e-3
        if ~RUN_TURBO
            plot(ax1,X(1,k+1), X(2,k+1),'go','MarkerFaceColor','g');
            plot(ax2,X(1,k+1), X(2,k+1),'go','MarkerFaceColor','g');
        end
        break;
    end
end

%% ---------- Final plot (turbo mode) ----------
if RUN_TURBO
    tl = tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
    ax1 = nexttile(tl,1); hold(ax1,'on'); axis(ax1,'equal'); grid(ax1,'on');
    plot(ax1,path.x, path.y,'b--','LineWidth',1.3);
    plot(ax1,X(1,1:k+1), X(2,1:k+1), 'r-','LineWidth',2);
    [vx,vy]=robotTriangle(X(1,k+1),X(2,k+1),X(3,k+1)); patch(ax1,'XData',vx,'YData',vy, ...
        'FaceColor',[0 0.4 1],'EdgeColor',[0 0.4 1],'FaceAlpha',0.45);
    title(ax1,sprintf('Final — t=%.1fs', k*Ts));
    xlabel(ax1,'x [m]'); ylabel(ax1,'y [m]');
    m=1.0; xlim(ax1,[min(path.x)-m, max(path.x)+m]); ylim(ax1,[min(path.y)-m, max(path.y)+m]);

    ax2 = nexttile(tl,2); hold(ax2,'on'); axis(ax2,'equal'); grid(ax2,'on');
    plot(ax2,path.x, path.y,'b--','LineWidth',1.3);
    d=decimateAccum; plot(ax2,X(1,1:d:k+1), X(2,1:d:k+1), 'r-','LineWidth',2);
    title(ax2,'Accumulated'); xlabel(ax2,'x [m]'); ylabel(ax2,'y [m]');
    xlim(ax2,[min(path.x)-m, max(path.x)+m]); ylim(ax2,[min(path.y)-m, max(path.y)+m]);
end

% quick metric
warm = round(1/Ts);
rms_err = compute_rms_to_path(X(:,1:k+1), path, warm);
fprintf('RMS path error (after warm-up) = %.3f m | steps=%d\n', rms_err, k);
end

%% ===================== Helpers =====================
function path = buildSplinePath(W, closed, N)
% Cubic spline through waypoints, with unwrapped tangent heading.
if closed, if norm(W(end,:) - W(1,:))>1e-9, W=[W;W(1,:)]; end, end
du = [0; cumsum( vecnorm(diff(W),2,2) )]; u_wp = du/du(end);
ppX = spline(u_wp, W(:,1)'); ppY = spline(u_wp, W(:,2)');
u  = linspace(0,1,N)'; x = ppval(ppX,u)'; y = ppval(ppY,u)';
dx = gradient(x,u); dy = gradient(y,u); d2x=gradient(dx,u); d2y=gradient(dy,u);
dsdu = hypot(dx,dy); s = cumtrapz(u,dsdu); s = s - s(1); s_end=s(end);
kappa = (dx.*d2y - dy.*d2x) ./ max(dsdu.^3, 1e-9);
theta = unwrap(atan2(dy,dx));
path.u=u; path.x=x; path.y=y; path.theta=theta; path.kappa=kappa; path.s=s; path.s_end=s_end;
end

function th = unwrapAround(thIn, th0)
th = unwrap([th0; thIn(:)]); th = th(2:end);
end

function [x,y,th,kap] = samplePath(path, s_query)
s=path.s; s_query = min(max(s_query,s(1)), path.s_end);
x=interp1(s,path.x,    s_query,'linear','extrap');
y=interp1(s,path.y,    s_query,'linear','extrap');
th=interp1(s,path.theta,s_query,'linear','extrap');
kap=interp1(s,path.kappa,s_query,'linear','extrap');
end

function idx = nearestIndexForward(xarr, yarr, xq, yq, idx_last, lookAhead)
% Nearest search only in a forward window [idx_last .. idx_last+lookAhead]
N = numel(xarr);
i0 = min(max(1,idx_last), N);
i1 = min(N, idx_last + lookAhead);
xr = xarr(i0:i1); yr = yarr(i0:i1);
[~,kmin] = min( (xr-xq).^2 + (yr-yq).^2 );
idx = i0 + kmin - 1;
end

function e = compute_rms_to_path(X, path, warm)
K=size(X,2); d=zeros(1,K);
for k=1:K
    [~,idx]=min( (path.x-X(1,k)).^2 + (path.y-X(2,k)).^2 );
    dx=X(1,k)-path.x(idx); dy=X(2,k)-path.y(idx);
    d(k)=hypot(dx,dy);
end
e = sqrt(mean( d(1, max(1,warm):end).^2 ));
end

function xk1 = ddrive_step_midpoint(xk, uk, Ts, L, aL, aR, xicr)
% Midpoint integrator for diff-drive.
% x=[x;y;th; w; vL; vR], u=[aL; aR]; enforce w=(vR-vL)/L at step end.
x = xk(1); y = xk(2); th = xk(3); vL = xk(5); vR = xk(6);
aLk= uk(1); aRk= uk(2);
vL1 = vL + Ts*aLk;  vR1 = vR + Ts*aRk;
vLmid = 0.5*(vL + vL1);  vRmid = 0.5*(vR + vR1);   % <-- fixed vR1
wmid  = (vRmid - vLmid)/L;
vxmid = 0.5*(aL*vLmid + aR*vRmid);
x1  = x  + Ts*( vxmid*cos(th) + xicr*sin(th)*wmid );
y1  = y  + Ts*( vxmid*sin(th) - xicr*cos(th)*wmid );
th1 = wrapToPiLocal( th + Ts*wmid );
w1  = (vR1 - vL1)/L;
xk1 = [x1; y1; th1; w1; vL1; vR1];
end

function ineq=v_bounds_ineq(X,vmin,vmax)
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

function [W, closed] = default_waypoints(id)
switch id
    case 1 % S-curve (open)
        W = [ -4 -1;  -2  1;   0 -1;   2  1;   4 -1 ];
        closed = false;
    case 2 % figure-eight (closed)
        a=2.0; tt=linspace(0,2*pi,161)'; x=a*sin(tt); y=a*sin(tt).*cos(tt);
        pick=1:10:numel(tt); W=[x(pick) y(pick)]; closed=true;
    otherwise, error('Unknown track_id. Use 1 or 2.');
end
end

%% ---------- Frenet-frame custom cost (with tiny slack penalty) ----------
function J = frenetCostLocal(X,U,e,~,YH)
% Stage cost over horizon in Frenet frame relative to YH = [x_r, y_r, th_r].
x  = X(1:end-1,1);  y  = X(1:end-1,2);  th = X(1:end-1,3);
xr = YH(:,1);       yr = YH(:,2);       thr = YH(:,3);

% lateral (cross-track) and heading errors
dx = x - xr;  dy = y - yr;
ey   = -sin(thr).*dx +  cos(thr).*dy;         % [m]
epsi = wrapToPiLocal(th - thr);               % [rad]

% smoothness
dU = diff(U);                                  % (Hp-1)×nu

% weights (tighten by raising w_ey; damp by raising w_du)
w_ey   = 1200;     % cross-track
w_epsi =   30;     % heading
w_u    = 0.002;    % input magnitude
w_du   = 0.015;    % input rate
w_slack= 1e-8;     % tiny penalty to silence "slack unused" notices

J = sum(w_ey*ey.^2 + w_epsi*epsi.^2) ...
  + w_u  * sum(sum(U.^2)) ...
  + w_du * sum(sum(dU.^2)) ...
  + w_slack * (e(:).' * e(:));
end

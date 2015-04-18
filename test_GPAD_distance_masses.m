%%
% Verify the implementation of dual-proximal gradient method with soft
% constraits on state and hard constraints on the control. The soft
% constraints on the state is given by a distance function from the
% feasible set. The spring mass system is taken as example for 
% simulations. 
clear all;
close all;
clc;
Nm=3; % Number of masses
Np=10; % prediction horizon 
T_sampling=0.5;
% system discription 
sys=system_masses(Nm,struct('Ts',T_sampling,'xmin', ...
    -4*ones(2*Nm,1), 'xmax', 4*ones(2*Nm,1), 'umin', -1.5*ones(Nm-1,1),'umax',...
    1.5*ones(Nm-1,1),'b', 0.1*ones(Nm+1,1)));

sys.Np=Np;

% cost functions 
V.Q=2*eye(sys.nx);
V.R=1*eye(sys.nu);

% soft constraints 
soft_ops.gamma_min=100;
soft_ops.gamma_max=100;
sys_dst=system_softvalues(sys,soft_ops);
% constraints
ny=size(sys.F,1);
F_temp=sys.F;
G_temp=sys.G;
g_temp=sys.g;

sys.F=cell(Np,1);
sys.G=cell(Np,1);
sys.g=cell(Np,1);

for i=1:Np
    sys.F{i}=F_temp;
    sys.G{i}=G_temp;
    sys.g{i}=g_temp;
end

sys.Ft=F_temp(1:2*sys.nx,1:sys.nx);
sys.gt=g_temp(1:2*sys.nx,1);
%% options of GPAD

ops_GPAD.steps=2000;
ops_GPAD.primal_inf=5e-3;
ops_GPAD.dual_gap=10e-3;
ops_GPAD.eq_feasibility=5e-3;

ops_GPAD.x0=2*rand(sys.nx,1)+0.4;

sigma=max([eig(V.Q);eig(V.R)]);
norm_dual_const=[max([norm([sys.F{1} sys.G{1}],2) norm(sys.Ft,2)]);...
    norm([sys_dst.F{1} sys_dst.G{1}],2)];

%% calculate the factor step matrices of GPAD
Ptree=GPAD_dynamic_formulation(sys,V);
Ptree_dst=GPAD_dynamic_formulation(sys_dst,V);
%% with distance soft constraints 
ops_GPAD.alpha=norm_dual_const(2)^2/sigma;
ops=ops_GPAD;
[Z_gpad,Y,details_GPAD]=GPAD_soft_distance(sys_dst,Ptree_dst,V,ops_GPAD);

%% with addtional variables
sigma_eps=max([eig(V_epsilon.Q);eig(V_epsilon.R);eig(V.Vf)]);
norm_dual_const_eps=max([norm([sys_dst.F{1} sys_dst.G{1}],2) norm(sys.Ft,2)]);
ops_GPAD_eps=ops_GPAD;
ops_GPAD_eps.alpha=norm_dual_const_eps^2/sigma_eps;
[Z_gpad_eps,Y_eps,details_GPAD_eps]=GPAD_soft_constraints_eps(sys_dst,Ptree_epsilon,V_epsilon,ops_GPAD_eps);
%%
yalmip_controller=yalmip_standard_mpc(sys,V);
[Z_yalmip,error]=yalmip_controller{ops_GPAD.x0};

%%
gpad_cost=0;
gpad_cost_eps=0;
yalmip_cost=0;
for i=1:sys.Np
    gpad_cost=gpad_cost+Z_gpad.U(:,i)'*V.R*Z_gpad.U(:,i)+Z_gpad.X(:,i)'*V.Q*Z_gpad.X(:,i);
    gpad_cost_eps=gpad_cost_eps+Z_gpad_eps.U(:,i)'*V_epsilon.R*Z_gpad_eps.U(:,i)...
        +Z_gpad_eps.X(:,i)'*V_epsilon.Q*Z_gpad.X(:,i);
    yalmip_cost=yalmip_cost+Z_yalmip{1,2}(:,i)'*V.R*Z_yalmip{1,2}(:,i)+...
        Z_yalmip{1,1}(:,i)'*V.Q*Z_yalmip{1,1}(:,i);
    for kk=1:sys.nx
        if(Z_gpad.X(kk,i+1)<sys.xs(kk))
            gpad_cost=gpad_cost+(sys.xs(kk)-Z_gpad.X(kk,i+1))'*V.Qs(kk,kk)*(sys.xs(kk)-Z_gpad.X(kk,i+1));
        end
    end
    for kk=1:sys.nx
        if(Z_yalmip{1,1}(kk,i+1)<sys.xs(kk))
            yalmip_cost=yalmip_cost+(sys.xs(kk)-Z_yalmip{1,1}(kk,i+1))'*V.Qs(kk,kk)*(sys.xs(kk)-Z_yalmip{1,1}(kk,i+1));
        end
    end
end
epsilon=max(0,kron(ones(1,Np-1),sys.xs)-Z_yalmip{1,1}(:,2:sys.Np));
gpad_cost=gpad_cost+Z_gpad.X(:,sys.Np+1)'*V.Vf*Z_gpad.X(:,sys.Np+1);
gpad_cost_eps=gpad_cost_eps+Z_gpad_eps.X(:,sys.Np+1)'*V_epsilon.Vf*Z_gpad_eps.X(:,sys.Np+1);
yalmip_cost=yalmip_cost+Z_yalmip{1,1}(:,sys.Np+1)'*V.Vf*Z_yalmip{1,1}(:,sys.Np+1);

%{
V1=V;
V1.Q=0*V.Q;
ny=zeros(sys.Np+2,1);
for i=1:sys.Np
    if(i==1)
        ny(i+1)=size(sys.F{i},1);
    else
        ny(i+1)=ny(i)+size(sys.F{i},1);
    end 
end
ny(sys.Np+2)=ny(sys.Np+1)+size(sys.Ft,1);
Y.y=zeros(ny(sys.Np+2),1);
dual_grad=dual_gradient_yalmip(sys,V1);
[Z,Q]=GPAD_dynamic_calculation(sys,Ptree,Y,ops_GPAD.x0);
Z_yalmip_grad=dual_grad{{ops_GPAD.x0,Y.y}};
%}


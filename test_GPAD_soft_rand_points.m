%% 
% Verify the implementation of dual-proximal gradient method with soft
% constraited system. The cost function depends on a threshold
% \mid(x-x_s)\mid_Q +u'Ru. The spring mass system is taken as example for 
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
sys.xs=1*ones(sys.nx,1);

% cost functions 
V.Q=2*eye(sys.nx);
V.Qs=0.2*eye(sys.nx);
V.R=1*eye(sys.nu);
% terminal cost 
V.Vf=dare(sys.A,sys.B,V.Q,V.R);

[sys_epsilon,V_epsilon]=system_soft_variable(sys,V);
%constraints
ny=size(sys.F,1);
F_temp=sys.F;
G_temp=sys.G;
g_temp=sys.g;

sys.F=cell(Np,1);
sys.G=cell(Np,1);
sys.g=cell(Np,1);

for i=1:Np
    if(i==1)
        sys.F{i}=F_temp;
        sys.G{i}=G_temp;
        sys.g{i}=g_temp;
    else
        sys.F{i}=[F_temp;eye(sys.nx)];
        sys.G{i}=[G_temp;zeros(sys.nx,sys.nu)];
        sys.g{i}=[g_temp;zeros(sys.nx,1)];
    end
end

% terminal consraints. 
sys.Ft=[eye(sys.nx);-eye(sys.nx)];
sys.gt=g_temp(1:2*sys.nx,1);
sys_epsilon.Ft=sys.Ft;
sys_epsilon.gt=sys.gt;

% test points 
test_points=98;
x_test=2*rand(sys.nx,100)+0.4;
gpad_cost=zeros(test_points,2);
gpad_cost_eps=zeros(test_points,2);
yalmip_cost=zeros(test_points,2);
%% yalmip implementation  optimizer
yalmip_controller=yalmip_implementation(sys,V);

%% calculate the factor step matrices of GPAD

Ptree=GPAD_dynamic_formulation(sys,V);
Ptree_epsilon=GPAD_dynamic_formulation(sys_epsilon,V_epsilon);
ops_GPAD.steps=2000;
ops_GPAD.primal_inf=1e-3;
ops_GPAD.dual_gap=10e-3;
ops_GPAD.eq_feasibility=1e-3;
%}
%% without additional variables
sigma=max([eig(V.Q); eig(V.Qs);eig(V.R);eig(V.Vf)]);
norm_dual_const=max([norm([sys.F{1} sys.G{1}],2) norm([sys.F{2} sys.G{2}],2) norm(sys.Ft,2)]);
ops_GPAD.alpha=norm_dual_const^2/sigma;
ops=ops_GPAD;

%% with addtional variables
sigma_eps=max([eig(V_epsilon.Q);eig(V_epsilon.R);eig(V.Vf)]);
norm_dual_const_eps=max([norm([sys_epsilon.F{1} sys_epsilon.G{1}],2) norm(sys.Ft,2)]);
ops_GPAD_eps=ops_GPAD;
ops_GPAD_eps.alpha=norm_dual_const_eps^2/sigma_eps;

%%
for i=1:test_points
    ops_GPAD.x0=x_test(:,i);
    ops_GPAD_eps.x0=x_test(:,i);
    [Z_yalmip,error]=yalmip_controller{ops_GPAD.x0};
    [Z_gpad,Y,details_GPAD]=GPAD_soft_constraints(sys,Ptree,V,ops_GPAD);
    [Z_gpad_eps,Y_eps,details_GPAD_eps]=GPAD_soft_constraints_eps(sys_epsilon,Ptree_epsilon,V_epsilon,ops_GPAD_eps);
    
    gpad_cost(i,2)=details_GPAD.iterate;
    gpad_cost_eps(i,2)=details_GPAD_eps.iterate;
    
    for j=1:sys.Np
        gpad_cost(i,1)=gpad_cost(i,1)+Z_gpad.U(:,j)'*V.R*Z_gpad.U(:,j)+Z_gpad.X(:,j)'*V.Q*Z_gpad.X(:,j);
        gpad_cost_eps(i,1)=gpad_cost_eps(i,1)+Z_gpad_eps.U(:,j)'*V_epsilon.R*Z_gpad_eps.U(:,j)...
            +Z_gpad_eps.X(:,j)'*V_epsilon.Q*Z_gpad.X(:,j);
        yalmip_cost(i,1)=yalmip_cost(i,1)+Z_yalmip{1,2}(:,j)'*V.R*Z_yalmip{1,2}(:,j)+...
            Z_yalmip{1,1}(:,j)'*V.Q*Z_yalmip{1,1}(:,j);
        for kk=1:sys.nx
            if(Z_gpad.X(kk,j+1)<sys.xs(kk))
                gpad_cost(i,1)=gpad_cost(i,1)+(sys.xs(kk)-Z_gpad.X(kk,j+1))'...
                    *V.Qs(kk,kk)*(sys.xs(kk)-Z_gpad.X(kk,j+1));
            end
        end
        for kk=1:sys.nx
            if(Z_yalmip{1,1}(kk,j+1)<sys.xs(kk))
                yalmip_cost(i,1)=yalmip_cost(i,1)+(sys.xs(kk)-Z_yalmip{1,1}(kk,j+1))'...
                    *V.Qs(kk,kk)*(sys.xs(kk)-Z_yalmip{1,1}(kk,j+1));
            end
        end
    end
    %epsilon=max(0,kron(ones(1,Np-1),sys.xs)-Z_yalmip{1,1}(:,2:sys.Np));
    gpad_cost(i,1)=gpad_cost(i,1)+Z_gpad.X(:,sys.Np+1)'*V.Vf*Z_gpad.X(:,sys.Np+1);
    gpad_cost_eps(i,1)=gpad_cost_eps(i,1)+Z_gpad_eps.X(:,sys.Np+1)'*V_epsilon.Vf*Z_gpad_eps.X(:,sys.Np+1);
    yalmip_cost(i,1)=yalmip_cost(i,1)+Z_yalmip{1,1}(:,sys.Np+1)'*V.Vf*Z_yalmip{1,1}(:,sys.Np+1);
end

%%
figure(1)
plot(gpad_cost(:,2))
hold all;
plot(gpad_cost_eps(:,2));
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


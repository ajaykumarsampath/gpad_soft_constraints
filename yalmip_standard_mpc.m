function [ controller] =yalmip_standard_mpc(sys,V)
% yalmip implementation of the cost function with smooth cost 

default_options=sdpsettings('solver','gurobi','verbose',0,'cachesolvers',1);

X=sdpvar(sys.nx,sys.Np+1);
U=sdpvar(sys.nu,sys.Np);
xint=sdpvar(sys.nx,1);

if(~isfield(sys,'Ft'))
    sys.Ft=zeros(1:2*sys.nx,1:sys.nx);
    sys.gt=ones(1:2*sys.nx,1);
end

if(~isfield(V,'Vf'))
    V.Vf=zeros(sys.nx);
end


J_obj=0;

constraints=(X(:,1)==xint);

for i=1:sys.Np
    constraints=constraints+(X(:,i+1)==sys.A*X(:,i)+sys.B*U(:,i));
    constraints=constraints+(sys.F{i}*X(:,i)+sys.G{i}*U(:,i)<=sys.g{i});
    J_obj=J_obj+U(:,i)'*V.R*U(:,i)+X(:,i)'*V.Q*X(:,i);
end

J_obj=J_obj+X(:,sys.Np+1)'*V.Vf*X(:,sys.Np+1);
constraints=constraints+(sys.Ft*X(:,sys.Np+1)<=sys.gt);
controller=optimizer(constraints,J_obj,default_options,{xint},{X,U,J_obj});
end



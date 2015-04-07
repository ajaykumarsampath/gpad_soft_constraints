function [ controller] =yalmip_implementation(sys,V)
% yalmip implementation of the cost function with smooth cost 

default_options=sdpsettings('solver','gurobi','verbose',0,'cachesolvers',1);

X=sdpvar(sys.nx,sys.Np+1);
epsilon=sdpvar(sys.nx,sys.Np-1);
U=sdpvar(sys.nu,sys.Np);
xint=sdpvar(sys.nx,1);
J_obj=0;
constraints=(X(:,1)==xint);

%{
ny=zeros(sys.Np+1,1);
for i=1:sys.Np
    if(i==1)
        ny(i)=size(sys.F{i},1);
    else
        ny(i)=ny(i-1)+size(sys.F{i},1);
    end 
end

ny(sys.Np+1,1)=ny(sys.Np,1)+size(sys.Ft,1);
%}
ny_s=size(sys.F{2},1)-sys.nx;
for i=1:sys.Np
    constraints=constraints+(X(:,i+1)==sys.A*X(:,i)+sys.B*U(:,i));
    if(i==1)
        constraints=constraints+(sys.F{i}*X(:,i)+sys.G{i}*U(:,i)<=sys.g{i});
    else
        constraints=constraints+(sys.F{i}(1:ny_s,:)*X(:,i)+sys.G{i}(1:ny_s,:)*U(:,i)<=sys.g{i}(1:ny_s,:));
        constraints=constraints+(epsilon(:,i-1)+X(:,i)>=sys.xs)+(epsilon(:,i-1)>=0);
    end
    if(i==1)
        J_obj=J_obj+U(:,i)'*V.R*U(:,i)+X(:,i)'*V.Q*X(:,i);
    else
        J_obj=J_obj+epsilon(:,i-1)'*V.Qs*epsilon(:,i-1)+U(:,i)'*V.R*U(:,i)+X(:,i)'*V.Q*X(:,i);
    end
    %+...(epsilon(:,i)<=sys.g{i}(1:sys.nx,:));
end
%{
J_obj=J_obj+epsilon(:,sys.Np+1)'*V.Vf*epsilon(:,sys.Np+1);
constraints=constraints+(sys.Ft(1:2*sys.nx,:)*X(:,sys.Np+1)<=sys.gt(1:2*sys.nx,1))+...
    (-epsilon(:,sys.Np+1)-X(:,sys.Np+1)<=-sys.xs)+(epsilon(:,sys.Np+1)>=0);
%}
J_obj=J_obj+X(:,sys.Np+1)'*V.Vf*X(:,sys.Np+1);
constraints=constraints+(sys.Ft(1:2*sys.nx,:)*X(:,sys.Np+1)<=sys.gt(1:2*sys.nx,1));
controller=optimizer(constraints,J_obj,default_options,{xint},{X,U,epsilon,J_obj});
end


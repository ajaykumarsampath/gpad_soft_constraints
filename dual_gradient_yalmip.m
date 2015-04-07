function [ dual_gradient ] = dual_gradient_yalmip( sys,V)
% dual_gradient_yalmip function create the yalmip optimiser 
% for calculating the dual gradient. 

default_options=sdpsettings('solver','gurobi','verbose',0,'cachesolvers',1);

ny=zeros(sys.Np+1,1);
for i=1:sys.Np
    if(i==1)
        ny(i)=size(sys.F{i},1);
    else
        ny(i)=ny(i-1)+size(sys.F{i},1);
    end 
end

ny(sys.Np+1)=ny(sys.Np)+size(sys.Ft,1);
xint=sdpvar(sys.nx,1);
X=sdpvar(sys.nx,sys.Np+1);
U=sdpvar(sys.nu,sys.Np);
Y=sdpvar(ny(sys.Np+1),1);
objective=0;
constraints=(X(:,1)==xint);

for i=1:sys.Np
    if(i==1)
        objective=objective+X(:,i)'*V.Q*X(:,i)+U(:,i)'*V.R*U(:,i)+Y(1:ny(i),1)'...
            *(sys.F{i}*X(:,i)+sys.G{i}*U(:,i)-sys.g{i});
    else
        objective=objective+X(:,i)'*V.Q*X(:,i)+U(:,i)'*V.R*U(:,i)+Y(ny(i-1)+1:ny(i),1)'...
            *(sys.F{i}*X(:,i)+sys.G{i}*U(:,i)-sys.g{i});
    end 

    constraints=constraints+(X(:,i+1)==sys.A*X(:,i)+sys.B*U(:,i));
end

objective=objective+X(:,sys.Np+1)'*V.Vf*X(:,sys.Np+1)+Y(ny(sys.Np)+1:ny(sys.Np+1),1)'...
    *(sys.Ft*X(:,sys.Np+1)-sys.gt);
dual_gradient=optimizer(constraints,objective,default_options,{xint,Y},{X,U,objective});
end


%% This function used to test the proximal operation of distance from two set

clc
close all;
clear all;

dim=3;
X.xs=-2+rand(dim,1);
X.xmax=2+rand(dim,1);
X.gamma_s=2;
X.gamma_max=6;
X.x=10*randn(dim,1);
%% calculation of projection 
x=sdpvar(dim,1);
proj=sdpvar(dim,1);
constraints=(proj<=X.xmax);
obj=norm(x-proj,2);
ops = sdpsettings('solver','gurobi','verbose',0,'cachesolvers',1);
projection_max=optimizer(constraints,obj,ops,{x},{proj});


X.proj_xmax=projection_max{X.x};

x=sdpvar(dim,1);
proj=sdpvar(dim,1);
constraints=(proj>=X.xs);
obj=norm(x-proj,2);
projection_min=optimizer(constraints,obj,ops,{x},{proj});

X.proj_xs=projection_min{X.x};

%% min the distance function
x=sdpvar(dim,1);
u=sdpvar(dim,1);
proj=sdpvar(dim,2);
obj_prox=X.gamma_max*norm(u-proj(:,1),2)+X.gamma_s*norm(u-proj(:,2),2)+0.5*norm(u-x,2)^2;
constraints=(proj(:,1)<=X.xmax)+(proj(:,2)>=X.xs);
constraints=constraints+(u-proj(:,1)>=0)+(u-proj(:,2)<=0);
prox_operator=optimizer(constraints,obj_prox,ops,{x},{u,proj,obj_prox});

L=prox_operator{X.x};
X.prox=L{1};
%% closed form proximal operator

X.prox_c=X.x;
if((min(X.x-X.xmax<0)==1)&&(min(X.x-X.xs>0)==1))
    for i=1:dim
        if(X.x(i,1)<X.xs(i,1))
            X.prox_c(i,1)=X.xs(i,1);
        elseif(X.x(i,1)>X.xmax(i,1))
            X.prox_c(i,1)=X.xmax(i,1);
        else
            X.prox_c(i,1)=X.x(i,1);
        end
    end
elseif(min(X.x-X.xmax<0)==1)
    if(norm(X.x-X.proj_xs,2)>X.gamma_s)
        X.prox_c=X.x+X.gamma_s*(X.proj_xs-X.x)/(norm(X.proj_xs-X.x,2));
    else
        X.prox_c=X.proj_xs;
    end
elseif(min(X.x-X.xs>0)==1)
    if(norm(X.x-X.proj_xmax,2)>X.gamma_max)
        X.prox_c=X.x+X.gamma_max*(X.proj_xmax-X.x)/(norm(X.x-X.proj_xmax,2));
    else
        X.prox_c=X.proj_xmax;
    end
else
    if(norm(X.prox_c-X.proj_xs,2)>X.gamma_s)
        X.prox_c=X.prox_c+X.gamma_s*(X.proj_xs-X.x)/(norm(X.proj_xs-X.prox_c,2));
    else
        X.prox_c=X.proj_xs;
    end
    pp=min(X.prox_c,X.xmax);
    if(norm(X.prox_c-pp,2)>X.gamma_max)
        X.prox_c=X.prox_c+X.gamma_max*(pp-X.prox_c)/(norm(X.prox_c-pp,2));
    else
        X.prox_c=pp;
    end
end
%max(abs(X.prox-X.prox_c))
assert(max(abs(X.prox-X.prox_c))<0.01)
%% closed form proximal operator
%p=[X.gamma_max X.gamma_s]/(X.gamma_max+X.gamma_s);
p=[1/2 1/2];
tt.y=zeros(dim,2);
i=1;
imax=1000;
while(i<imax)
    tt.gamma_max=X.gamma_max/p(1);
    tt.gamma_s=X.gamma_s/p(2);
    %tt.x(:,1)=X.x-tt.y(:,1)/p(1);
    tt.x(:,1)=X.x-tt.y(:,1);
    tt.proj_max=min(tt.x(:,1),X.xmax);
    if(norm(tt.x(:,1)-tt.proj_max,2)>tt.gamma_max)
        tt.x(:,1)=tt.x(:,1)+tt.gamma_max*(tt.proj_max-tt.x(:,1))/(norm(tt.x(:,1)-tt.proj_max,2));
    else
        tt.x(:,1)=tt.proj_max;
    end  
    %tt.x(:,2)=X.x-tt.y(:,2)/p(2);
    tt.x(:,2)=X.x-tt.y(:,2);
    tt.proj_s=max(tt.x(:,2),X.xs);
    if(norm(tt.x(:,2)-tt.proj_s,2)>tt.gamma_s)
        tt.x(:,2)=tt.x(:,2)+tt.gamma_s*(tt.proj_s-tt.x(:,2))/(norm(tt.x(:,2)-tt.proj_s,2));
    else
        tt.x(:,2)=tt.proj_s;
    end
    tt.y=tt.y+tt.x-kron([1 1],mean(tt.x')');
    if(max(abs(tt.x(:,1)-tt.x(:,2)))<0.001)
        iter=i;
        i=imax+200;
        %i=i+1;
    else
        i=i+1;
    end
end
if(min((tt.x(:,1)-X.xmax))<0.001)
    t=tt.x(:,1);
    t_proj_s=max(t,X.xs);
    if(norm(t-t_proj_s,2)>X.gamma_s)
        t=t+X.gamma_s*(t_proj_s-t)/(norm(t-t_proj_s,2));
    else
        t=t_proj_s;
    end 
end
    
tt.proj_max=min(tt.x(:,1),X.xmax);
tt.proj_s=max(tt.x(:,2),X.xs);
tt.obj_value=X.gamma_max*norm(tt.x(:,1)-tt.proj_max,2)+X.gamma_s*norm(tt.x(:,2)-tt.proj_s,2)...
    +0.5*norm(tt.x(:,1)-X.x,2)^2;
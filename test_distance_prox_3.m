%% This function used to test the proximal operation of distance from two set

clc
close all;
clear all;

dim=3;
X.xs=rand(dim,1)-1;
X.xmax=rand(dim,1)+3;
X.xmin=rand(dim,1)-3;
X.gamma_s=2;
X.gamma_max=2;
X.gamma_min=2;
X.x=3*randn(dim,1)-20;
%X.x=[-0.5;-0.5;-2.05];
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

x=sdpvar(dim,1);
proj=sdpvar(dim,1);
constraints=(proj>=X.xmin);
obj=norm(x-proj,2);
projection_min=optimizer(constraints,obj,ops,{x},{proj});

X.proj_xmin=projection_min{X.x};
%% min the distance function
x=sdpvar(dim,1);
u=sdpvar(dim,1);
proj=sdpvar(dim,3);
obj_prox=X.gamma_max*norm(u-proj(:,1),2)+X.gamma_s*norm(u-proj(:,2),2)...
    +X.gamma_min*norm(u-proj(:,3),2)+0.5*norm(u-x,2)^2;
constraints=(proj(:,1)<=X.xmax)+(proj(:,2)>=X.xs)+(proj(:,3)>=X.xmin);
constraints=constraints+(u-proj(:,1)>=0)+(u-proj(:,2)<=0)+(u-proj(:,3)<=0);
prox_operator=optimizer(constraints,obj_prox,ops,{x},{u,proj,obj_prox});

L=prox_operator{X.x};
X.prox=L{1};
L{1,4}=L{1,2}-kron(ones(1,3),L{1});

% closed form proximal operator
C1=min(X.x-X.xmax<0);
C2=min(X.x-X.xs>0);
C3=min(X.x-X.xmin>0);

distance=[norm(X.x-X.proj_xmax,2);norm(X.x-X.proj_xs,2);norm(X.x-X.proj_xmin,2)];
t=[X.gamma_max/distance(1);X.gamma_s/distance(2);X.gamma_min/distance(3)];
%% calculation of prox for 3 sets 
%
p=[1/3 1/3 1/3];
%p=[X.gamma_max X.gamma_s X.gamma_min]/(X.gamma_min+X.gamma_s+X.gamma_max);
%p=[0 X.gamma_s X.gamma_min]/(X.gamma_min+X.gamma_s);
tt.y=zeros(dim,3);
%tt.x=[X.x X.x];
i=1;
imax=100;
while(i<imax)
    tt.gamma_max=X.gamma_max/p(1);
    tt.gamma_s=X.gamma_s/p(2);
    tt.gamma_min=X.gamma_min/p(3);
    
    tt.x(:,1)=X.x-tt.y(:,1);
    tt.proj_min=max(tt.x(:,1),X.xmin);
    if(norm(tt.x(:,1)-tt.proj_min,2)>tt.gamma_min)
        tt.x(:,1)=tt.x(:,1)+tt.gamma_min*(tt.proj_min-tt.x(:,1))/(norm(tt.x(:,1)-tt.proj_min,2));
    else
        tt.x(:,1)=tt.proj_min;
    end  
    
    tt.x(:,2)=X.x-tt.y(:,2);
    tt.proj_s=max(tt.x(:,2),X.xs);
    if(norm(tt.x(:,2)-tt.proj_s,2)>tt.gamma_s)
        tt.x(:,2)=tt.x(:,2)+tt.gamma_s*(tt.proj_s-tt.x(:,2))/(norm(tt.x(:,2)-tt.proj_s,2));
    else
        tt.x(:,2)=tt.proj_s;
    end
    
    tt.x(:,3)=X.x-tt.y(:,3);
    tt.proj_max=min(tt.x(:,3),X.xmax);
    if(norm(tt.x(:,3)-tt.proj_max,2)>tt.gamma_max)
        tt.x(:,3)=tt.x(:,3)+tt.gamma_max*(tt.proj_max-tt.x(:,3))/(norm(tt.x(:,3)-tt.proj_max,2));
    else
        tt.x(:,3)=tt.proj_max;
    end 
    tt.y=tt.y+tt.x-kron([1 1 1],mean(tt.x')');
    if(max(abs(tt.x(:,1)-tt.x(:,2)))<0.001 && max(abs(tt.x(:,2)-tt.x(:,3)))<0.001)
        iter=i;
        i=imax+200;
        %i=i+1;
    else
        i=i+1;
    end
end
if(min((tt.x(:,1)-X.xmin))<0.001)
    t=tt.x(:,1);
    t_proj_s=max(t,X.xs);
    if(norm(t-t_proj_s,2)>X.gamma_s)
        t=t+X.gamma_s*(t_proj_s-t)/(norm(t-t_proj_s,2));
    else
        t=t_proj_s;
    end 
end

tt.proj_min=max(tt.x(:,1),X.xmin);
tt.proj_s=max(tt.x(:,2),X.xs);
tt.proj_max=min(tt.x(:,3),X.xmax);
tt.obj_value=X.gamma_max*norm(tt.x(:,3)-tt.proj_max,2)+X.gamma_min*norm(tt.x(:,1)-tt.proj_min,2)...
    +X.gamma_s*norm(tt.x(:,2)-tt.proj_s,2)+0.5*norm(tt.x(:,1)-X.x,2)^2;

tt.proj_max=min(X.prox,X.xmax);
tt.proj_min=max(X.prox,X.xmin);
tt.proj_s=max(X.prox,X.xs);
obj_value=X.gamma_max*norm(X.prox-tt.proj_max,2)+X.gamma_min*norm(X.prox-tt.proj_min,2)...
    +X.gamma_s*norm(X.prox-tt.proj_s,2)+0.5*norm(X.prox-X.x,2)^2;
%% calculation of prox for 2 sets 
%{

p=[X.gamma_min X.gamma_s]/(X.gamma_min+X.gamma_s);
tt.y=zeros(dim,2);
i=1;
imax=1000;
while(i<imax)
    tt.gamma_min=X.gamma_min/p(1);
    tt.gamma_s=X.gamma_s/p(2);
    %tt.x(:,1)=X.x-tt.y(:,1)/p(1);
    tt.x(:,1)=X.x-tt.y(:,1);
    tt.proj_min=max(tt.x(:,1),X.xmin);
    if(norm(tt.x(:,1)-tt.proj_min,2)>tt.gamma_min)
        tt.x(:,1)=tt.x(:,1)+tt.gamma_min*(tt.proj_min-tt.x(:,1))/(norm(tt.x(:,1)-tt.proj_min,2));
    else
        tt.x(:,1)=tt.proj_min;
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
if(min((tt.x(:,1)-X.xmin))<0.001)
    t=tt.x(:,1);
    t_proj_s=max(t,X.xs);
    if(norm(t-t_proj_s,2)>X.gamma_s)
        t=t+X.gamma_s*(t_proj_s-t)/(norm(t-t_proj_s,2));
    else
        t=t_proj_s;
    end 
end
    
tt.proj_min=max(tt.x(:,1),X.xmin);
tt.proj_s=max(tt.x(:,2),X.xs);
tt.obj_value=X.gamma_min*norm(tt.x(:,1)-tt.proj_min,2)+X.gamma_s*norm(tt.x(:,2)-tt.proj_s,2)...
    +0.5*norm(tt.x(:,1)-X.x,2)^2;

%tt.proj_min=max(X.prox,X.xmin);
%tt.proj_s=max(X.prox,X.xs);
%obj_value=X.gamma_min*norm(X.prox-tt.proj_min,2)+X.gamma_s*norm(X.prox-tt.proj_s,2)...
%    +0.5*norm(X.prox(:,1)-X.x,2)^2;
%}
%% 
%{
X.x=1*randn(dim,1)+3;
X.y=3*randn(dim,1);
x=sdpvar(dim,1);
y=sdpvar(dim,1);
t=sdpvar(dim,1);
proj=sdpvar(dim,1);
constraints=(proj<=X.xmax)+(x-proj>=0);
obj=X.gamma_max*norm(x-proj,2)+y'*x+0.25*norm(x-t,2)^2;
projection_max=optimizer(constraints,obj,ops,{y,t},{x,proj,obj});

[LL,error]=projection_max{{X.y,X.x}};
%LL{1,4}=min(LL{1,1},X.xmax);
%}
%% calculation of prox for 2 sets 
%{
clear tt
J=zeros(1,3);
tt.x=[X.x X.x];
tt.gamma_min=X.gamma_min;
tt.gamma_s=X.gamma_s;
i=1;
imax=100;
while(i<imax)
    tt.proj_min=max(tt.x(:,1),X.xmin);
    if(norm(tt.x(:,1)-tt.proj_min,2)>tt.gamma_min)
        tt.x(:,2)=tt.x(:,1)+tt.gamma_min*(tt.proj_min-tt.x(:,1))/(norm(tt.x(:,1)-tt.proj_min,2));
    else
        tt.x(:,2)=tt.proj_min;
    end
    tt.proj_s=max(tt.x(:,2),X.xs);
    if(norm(tt.x(:,2)-tt.proj_s,2)>tt.gamma_s)
        tt.x(:,1)=tt.x(:,2)+tt.gamma_s*(tt.proj_s-tt.x(:,2))/(norm(tt.x(:,2)-tt.proj_s,2));
    else
        tt.x(:,1)=tt.proj_s;
    end
    if(max(abs(tt.x(:,1)-tt.x(:,2)))<0.001)
        iter=i;
        i=imax+200;
        %i=i+1;
    else
        i=i+1;
    end
end
tt.proj_min=max(tt.x(:,1),X.xmin);
tt.proj_s=max(tt.x(:,2),X.xs);
tt.obj_value=tt.gamma_min*norm(tt.x(:,1)-tt.proj_min,2)+tt.gamma_s*norm(tt.x(:,2)-tt.proj_s,2)...
    +0.5*norm(tt.x(:,1)-X.x,2)^2;
%% calculation of prox for 2 sets 
clear tt
J=zeros(1,3);
tt.x=X.x;
tt.gamma_min=X.gamma_min;
tt.gamma_s=X.gamma_s;
i=1;
imax=100;
while(i<imax)
    tt.proj_min=max(tt.x,X.xmin);
    if(norm(tt.x-tt.proj_min,2)>tt.gamma_min)
        tt.x=tt.x+tt.gamma_min*(tt.proj_min-tt.x)/(norm(tt.x-tt.proj_min,2));
    else
        tt.x=tt.proj_min;
    end
    tt.proj_s=max(tt.x,X.xs);
    if(norm(tt.x-tt.proj_s,2)>tt.gamma_s)
        tt.x=tt.x+tt.gamma_s*(tt.proj_s-tt.x)/(norm(tt.x-tt.proj_s,2));
    else
        tt.x=tt.proj_s;
    end
    i=i+1;
end
tt.proj_min=max(tt.x(:,1),X.xmin);
tt.proj_s=max(tt.x(:,2),X.xs);
tt.obj_value=tt.gamma_min*norm(tt.x(:,1)-tt.proj_min,2)+tt.gamma_s*norm(tt.x(:,2)-tt.proj_s,2)...
    +0.5*norm(tt.x(:,1)-X.x,2)^2;
%}
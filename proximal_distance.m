function [ zq,prox_details] = proximal_distance(yq,sys_dst)

% proximal_distance cacluate the proximal of the yq with 
% from the set [xmin xmax] with parameters gamma_min, gamma_max 
% specified.

p=[1/2 1/2];
dim=size(yq,1);
prox_details.y=zeros(dim,2);
prox_details.x=zeros(dim,2);
i=1;
imax=1000;
while(i<imax)
    prox_details.gamma_max=sys_dst.gamma_max/p(1);
    prox_details.gamma_min=sys_dst.gamma_min/p(2);
    
    prox_details.x(:,1)=yq-prox_details.y(:,1);
    prox_details.proj_max=min(prox_details.x(:,1),sys_dst.xmax);
    
    if(norm(prox_details.x(:,1)-prox_details.proj_max,2)>prox_details.gamma_max)
        prox_details.x(:,1)=prox_details.x(:,1)+prox_details.gamma_max*...
            (prox_details.proj_max-prox_details.x(:,1))/(norm(prox_details.x(:,1)-prox_details.proj_max,2));
    else
        prox_details.x(:,1)=prox_details.proj_max;
    end  
    
    prox_details.x(:,2)=yq-prox_details.y(:,2);
    prox_details.proj_min=max(prox_details.x(:,2),sys_dst.xmin);
    if(norm(prox_details.x(:,2)-prox_details.proj_min,2)>prox_details.gamma_min)
        prox_details.x(:,2)=prox_details.x(:,2)+prox_details.gamma_min*...
            (prox_details.proj_min-prox_details.x(:,2))/(norm(prox_details.x(:,2)-prox_details.proj_min,2));
    else
        prox_details.x(:,2)=prox_details.proj_min;
    end
    prox_details.y=prox_details.y+prox_details.x-kron([1 1],mean(prox_details.x')');
    if(max(abs(prox_details.x(:,1)-prox_details.x(:,2)))<0.001)
        prox_details.iter=i;
        i=imax+200;
        %i=i+1;
    else
        i=i+1;
    end
end

prox_details.proj_max=min(prox_details.x(:,1),sys_dst.xmax);
prox_details.proj_min=max(prox_details.x(:,2),sys_dst.xmin);
prox_details.obj_value=sys_dst.gamma_max*norm(prox_details.x(:,1)-prox_details.proj_max,2)...
    +sys_dst.gamma_min*norm(prox_details.x(:,2)-prox_details.proj_min,2)...
    +0.5*norm(prox_details.x(:,1)-yq,2)^2;

zq=prox_details.x(:,1);
end


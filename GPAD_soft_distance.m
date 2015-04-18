function [ Z,Y,details] = GPAD_soft_distance(sys_dst,Ptree_dst,V,ops)
% This function is implements the GPAD algorithm to solve an optimization
% with . The inputs are the system dynamics,
% and the matrices calculated in the factor step.

%%
% Initalizing the dual varibables
ny=zeros(sys_dst.Np+1,2);
for i=1:sys_dst.Np
    ny(i+1,1)=ny(i,1)+sys_dst.nx;
    ny(i+1,2)=ny(i,2)+size(sys_dst.F{i,1},1)-sys_dst.nx;
end 

Y.y0=zeros(ny(sys_dst.Np+1,1)+ny(sys_dst.Np+1,2),1);
Y.y1=zeros(ny(sys_dst.Np+1,1)+ny(sys_dst.Np+1,2),1);

prm_fes.dist_const=zeros(ny(sys_dst.Np+1,1),1);
prm_fes.hard_const=zeros(ny(sys_dst.Np+1,2),1);


dual_grad=prm_fes;

dual_variable.dist_const=zeros(ny(sys_dst.Np+1,1),1);
dual_variable.hard_const=zeros(ny(sys_dst.Np+1,2),1);


epsilon_prm=1;

g_nodes.dist_const=zeros(ny(sys_dst.Np+1,1),1);
g_nodes.hard_const=zeros(ny(sys_dst.Np+1,2),1);

for i=1:sys_dst.Np
    g_nodes.dist_const(ny(i,1)+1:ny(i+1,1),1)=sys_dst.g{i}(1:sys_dst.nx,1);
    g_nodes.hard_const(ny(i,2)+1:ny(i+1,2),1)=sys_dst.g{i}(sys_dst.nx+1:end,1);
end
theta=[1 1]';
tic
j=1;
details.term_crit=zeros(1,4);
%%
while(j<ops.steps)
    % step 1: accelerated step
    W.y=Y.y1+theta(2)*(1/theta(1)-1)*(Y.y1-Y.y0);
    
    % step 2: argmin of the lagrangian using dynamic programming
    [Z,Q]=GPAD_dynamic_calculation(sys_dst,Ptree_dst,W,ops.x0);
    
    % details.Z{j}=Z;
    % step 3: proximal step [y_p y_q]'
    % calculate the primal infeasibility conditions.
    % proximal update of y_p is projection on positive quadrant;
    Y.y0=Y.y1;
    yq=zeros(sys_dst.nx*sys_dst.Np,1);
    
    for i=1:sys_dst.Np
        
        prm_fes.hard_const(ny(i,2)+1:ny(i+1,2),1)=sys_dst.G{i}(sys_dst.nx+1:end,:)*Z.U(:,i);
        
        dual_grad.dist_const(ny(i,1)+1:ny(i+1,1),1)=Z.X(:,i+1);
        dual_grad.hard_const(ny(i,2)+1:ny(i+1,2),1)=...
            prm_fes.hard_const(ny(i,2)+1:ny(i+1,2),1)-g_nodes.hard_const(ny(i,2)+1:ny(i+1,2),1);
        
        dual_variable.dist_const(ny(i,1)+1:ny(i+1,1),1)=W.y(ny(i,1)+ny(i,2)+1:ny(i,2)+ny(i+1,1),1);
        dual_variable.hard_const(ny(i,2)+1:ny(i+1,2),1)=W.y(ny(i,2)+ny(i+1,1)+1:ny(i+1,1)+ny(i+1,2),1);
        
        Y.y1(ny(i,2)+ny(i+1,1)+1:ny(i+1,1)+ny(i+1,2),1)=max(0,W.y(ny(i,2)+ny(i+1,1)+1:ny(i+1,1)+ny(i+1,2),1)...
            +ops.alpha*dual_grad.hard_const(ny(i,2)+1:ny(i+1,2),1));
        
        yq((i-1)*sys_dst.nx+1:i*sys_dst.nx,1)=W.y(ny(i,1)+ny(i,2)+1:ny(i,2)+ny(i+1,1),1)/ops.alpha+...
            Z.X(:,i+1);
        
    end
    
    [zq,prox_details]=proximal_distance(yq,sys_dst);
    
    for i=1:sys_dst.Np
        Y.y1(ny(i,1)+ny(i,2)+1:ny(i,2)+ny(i+1,1),1)=W.y(ny(i,1)+ny(i,2)+1:ny(i,2)+ny(i+1,1),1)...
            +ops.alpha*(dual_grad.dist_const(ny(i,1)+1:ny(i+1,1),1)-zq(ny(i,1)+1:ny(i+1,1),1));
    end
    
    iter=j;
    %details.prm_cst(iter)=0;%primal cost;
    %details.dual_cst(iter)=0;% dual cost;
    
    %
    %termination criteria
    if(j==1)
        prm_avg_next.hard_const=prm_fes.hard_const;
        epsilon_prm_avg=max(prm_fes.hard_const-g_nodes.hard_const);
    else
        prm_avg_next.hard_const=(1-theta(2))*prm_avg_next.hard_const+theta(2)*prm_fes.hard_const;
        epsilon_prm_avg=max(prm_avg_next.hard_const-g_nodes.hard_const);
    end
    
    %if epsilon_prm_avg<=ops.primal_inf %average_primal feasibility less
        %details.term_crit(1,2)=1;
        %details.iterate=j;
        %j=10*ops.steps;
    %else
        details.eq_feasibility(iter)=max(abs(dual_grad.dist_const-zq));
        if(max(abs(dual_grad.dist_const-zq))<ops.eq_feasibility)
            epsilon_prm=max(prm_fes.hard_const-g_nodes.hard_const);
            if(epsilon_prm<=ops.primal_inf) % primal feasibility of the iterate
                if (min(dual_variable.hard_const)>0)
                    sum=-dual_variable.hard_const'*(prm_fes.hard_const-g_nodes.hard_const);
                    if sum<=ops.dual_gap %condition 29. dual gap
                        details.term_crit(1,2)=1;
                        details.iterate=j;
                        j=10*ops.steps;
                    else
                        prm_cst=0;%primal cost;
                        for i=1:sys_dst.Np
                            prm_cst=prm_cst+Z.X(:,i)'*V.Q*Z.X(:,i)+Z.U(:,i)'*V.R*Z.U(:,i);
                        end
                        if sum<=ops.dual_gap*prm_cst/(1+ops.dual_gap) %condition 30 dual gap
                            details.term_crit(1,3)=1;
                            details.iterate=j;
                            j=10*ops.steps;
                        else
                            %step 4: theta update
                            theta(1)=theta(2);
                            theta(2)=(sqrt(theta(1)^4+4*theta(1)^2)-theta(1)^2)/2;
                            j=j+1;
                        end
                    end
                else
                    prm_cst=0;%primal cost;
                    for i=sys_dst.Np
                        prm_cst=prm_cst+Z.X(:,i)'*V.Q*Z.X(:,i)+Z.U(:,i)'*V.R*Z.U(:,i);
                    end
                    dual_cst=prm_cst+dual_variable.hard_const'*(prm_fes.hard_const-g_nodes.hard_const);
                    if (-dual_cst<=ops.dual_gap*max(dual_cst,1)) %condtion 27 (dual gap)
                        details.term_crit(1,4)=1;
                        details.iterate=j;
                        j=10*ops.steps;
                    else
                        %step 4: theta update
                        theta(1)=theta(2);
                        theta(2)=(sqrt(theta(1)^4+4*theta(1)^2)-theta(1)^2)/2;
                        j=j+1;
                    end
                end
            else
                %step 4: theta update
                theta(1)=theta(2);
                theta(2)=(sqrt(theta(1)^4+4*theta(1)^2)-theta(1)^2)/2;
                j=j+1;
            end
        else
            theta(1)=theta(2);
            theta(2)=(sqrt(theta(1)^4+4*theta(1)^2)-theta(1)^2)/2;
            j=j+1;
        end
        
    %end
    details.epsilon_prm_avg(iter)=epsilon_prm_avg;
    details.epsilon_prm(iter)=epsilon_prm;
    %}
    %{
    theta(1)=theta(2);
    theta(2)=(sqrt(theta(1)^4+4*theta(1)^2)-theta(1)^2)/2;
    j=j+1;
    %}
end
%%
details.gpad_solve=toc;
details.W=W;
details.yq=yq;
details.prm_fes=prm_fes;
details.dual_grad=dual_grad;
details.prox_details=prox_details;
%details.epsilon_prm_avg= epsilon_prm_avg;
%details.epsilon_prm=epsilon_prm;
end


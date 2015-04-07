function [ Z,Y,details] = GPAD_soft_constraints_eps(sys_epsilon,Ptree_epsilon,V_epsilon,ops_GPAD_eps)

%This function is implements the GPAD algorithm to solve an optimization
%problem with non-smooth cost function. The inputs are the system dynamics,
% and the matrices calculated in the factor step.

%%
% Initalizing the dual varibables
ny=zeros(sys_epsilon.Np+2,1);
for i=1:sys_epsilon.Np
    ny(i+1,1)=ny(i,1)+size(sys_epsilon.F{i},1);
end

ny(sys_epsilon.Np+2,1)=ny(sys_epsilon.Np+1,1)+size(sys_epsilon.Ft,1);

Y.y0=zeros(ny(sys_epsilon.Np+2,1),1);
Y.y1=zeros(ny(sys_epsilon.Np+2,1),1);
prm_fes.hard_const=zeros(ny(sys_epsilon.Np+2,1),1);

dual_grad=prm_fes;
dual_variable.hard_const=zeros(ny(sys_epsilon.Np+2,1),1);

epsilon_prm=1;

g_nodes.hard_const=zeros(ny(sys_epsilon.Np+2,1),1);

for i=1:sys_epsilon.Np
    g_nodes.hard_const(ny(i,1)+1:ny(i+1,1),1)=sys_epsilon.g{i}(1:ny(i+1,1)-ny(i,1),1);
end

g_nodes.hard_const(ny(sys_epsilon.Np+1,1)+1:ny(sys_epsilon.Np+2,1),1)=sys_epsilon.gt;

theta=[1 1]';
tic
j=1;
details.term_crit=zeros(1,4);
%%
while(j<ops_GPAD_eps.steps)
    % step 1: accelerated step
    W.y=Y.y1+theta(2)*(1/theta(1)-1)*(Y.y1-Y.y0);
    
    % step 2: argmin of the lagrangian using dynamic programming
    [Z,Q]=GPAD_dynamic_calculation(sys_epsilon,Ptree_epsilon,W,ops_GPAD_eps.x0);
    
    % details.Z{j}=Z;
    % step 3: proximal step [y_p y_q]'
    % calculate the primal infeasibility conditions.
    % proximal update of y_p is projection on positive quadrant;
    Y.y0=Y.y1;
    for i=1:sys_epsilon.Np+1
        if(i==sys_epsilon.Np+1)
            prm_fes.hard_const(ny(i,1)+1:ny(i+1,1),1)=sys_epsilon.Ft*Z.X(:,i);
            dual_grad.hard_const(ny(i,1)+1:ny(i+1,1),1)=prm_fes.hard_const(ny(i,1)+1:ny(i+1,1),1)...
                -g_nodes.hard_const(ny(i,1)+1:ny(i+1,1),1);
            dual_variable.hard_const(ny(i,1)+1:ny(i+1,1),1)=W.y(ny(i,1)+1:ny(i+1,1),1);
            Y.y1(ny(i,1)+1:ny(i+1,1),1)=max(0,W.y(ny(i,1)+1:ny(i+1,1),1)...
                +ops_GPAD_eps.alpha*dual_grad.hard_const(ny(i,1)+1:ny(i+1,1),1));
        else
            if(i==1)
                prm_fes.hard_const(ny(i,1)+1:ny(i+1,1),1)=sys_epsilon.F{i}*Z.X(:,i)+sys_epsilon.G{i}*Z.U(:,i);
                
                dual_grad.hard_const(ny(i,1)+1:ny(i+1,1),1)=...
                    prm_fes.hard_const(ny(i,1)+1:ny(i+1,1),1)-g_nodes.hard_const(ny(i,1)+1:ny(i+1,1),1);
                Y.y1(ny(i,1)+1:ny(i+1,1),1)=max(0,W.y(ny(i,1)+1:ny(i+1,1),1)+...
                    ops_GPAD_eps.alpha*dual_grad.hard_const(ny(i,1)+1:ny(i+1,1),1));
                dual_variable.hard_const(ny(i,1)+1:ny(i+1,1),1)=W.y(ny(i,1)+1:ny(i+1,1),1);
            else
                prm_fes.hard_const(ny(i,1)+1:ny(i+1,1),1)=sys_epsilon.F{i}*Z.X(:,i)+sys_epsilon.G{i}*Z.U(:,i);
                
                dual_grad.hard_const(ny(i,1)+1:ny(i+1,1),1)=prm_fes.hard_const(ny(i,1)+1:ny(i+1,1),1)...
                    -g_nodes.hard_const(ny(i,1)+1:ny(i+1,1),1);
                
                Y.y1(ny(i,1)+1:ny(i+1,1),1)=max(0,W.y(ny(i,1)+1:ny(i+1,1),1)+...
                    ops_GPAD_eps.alpha*dual_grad.hard_const(ny(i,1)+1:ny(i+1,1),1));
                dual_variable.hard_const(ny(i,1)+1:ny(i+1,1),1)=W.y(ny(i,1)+1:ny(i+1,1),1);
            end
        end
    end
    
    iter=j;
    details.prm_cst(iter)=0;%primal cost;
    details.dual_cst(iter)=0;% dual cost;
    
    %
    %termination criteria
    if(j==1)
        prm_avg_next.hard_const=prm_fes.hard_const;
        epsilon_prm_avg=max(prm_fes.hard_const-g_nodes.hard_const);
    else
        prm_avg_next.hard_const=(1-theta(2))*prm_avg_next.hard_const+theta(2)*prm_fes.hard_const;
        epsilon_prm_avg=max(prm_avg_next.hard_const-g_nodes.hard_const);
    end
    
    if epsilon_prm_avg<=ops_GPAD_eps.primal_inf %average_primal feasibility less
        details.term_crit(1,2)=1;
        details.iterate=j;
        j=10*ops.steps;
    else
        epsilon_prm=max(prm_fes.hard_const-g_nodes.hard_const);
        if(epsilon_prm<=ops_GPAD_eps.primal_inf) % primal feasibility of the iterate
            if (min(dual_variable.hard_const)>0)
                sum=-dual_variable.hard_const'*(prm_fes.hard_const-g_nodes.hard_const);
                if sum<=ops_GPAD_eps.dual_gap %condition 29. dual gap
                    details.term_crit(1,2)=1;
                    details.iterate=j;
                    j=10*ops_GPAD_eps.steps;
                else
                    prm_cst=0;%primal cost;
                    for i=1:sys_epsilon.Np
                        prm_cst=prm_cst+Z.U(:,i)'*V_epsilon.R*Z.U(:,i)+Z.X(:,i)'*V_epsilon*Z.X(:,i);
                    end
                    if sum<=ops_GPAD_eps.dual_gap*prm_cst/(1+ops_GPAD_eps.dual_gap) %condition 30 dual gap
                        details.term_crit(1,3)=1;
                        details.iterate=j;
                        j=10*ops_GPAD_eps.steps;
                    else
                        %step 4: theta update
                        theta(1)=theta(2);
                        theta(2)=(sqrt(theta(1)^4+4*theta(1)^2)-theta(1)^2)/2;
                        j=j+1;
                    end
                end
            else
                prm_cst=0;%primal cost;
                for i=sys_epsilon.Np
                    prm_cst=prm_cst+Z.U(:,i)'*V_epsilon.R*Z.U(:,i);
                end
                dual_cst=prm_cst+dual_variable.hard_const'*(prm_fes.hard_const-g_nodes.hard_const);
                if (-dual_cst<=ops_GPAD_eps.dual_gap*max(dual_cst,1)) %condtion 27 (dual gap)
                    details.term_crit(1,4)=1;
                    details.iterate=j;
                    j=10*ops_GPAD_eps.steps;
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
details.prm_fes=prm_fes;
details.dual_grad=dual_grad;
%details.epsilon_prm_avg= epsilon_prm_avg;
%details.epsilon_prm=epsilon_prm;
end




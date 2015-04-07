function [ Z,Y,details] = GPAD_soft_constraints(sys,Ptree,V,ops)
%This function is implements the GPAD algorithm to solve an optimization
%problem with non-smooth cost function. The inputs are the system dynamics,
% and the matrices calculated in the factor step.

%%
Np=sys.Np;%prediction horizon

% Initalizing the dual varibables
ny=zeros(sys.Np+2,2);
for i=1:sys.Np
    if(i==1)
        ny(i+1,1)=size(sys.F{i},1);
        ny(i+1,2)=0;
    else
        ny(i+1,1)=ny(i,1)+size(sys.F{i},1)-sys.nx;
        ny(i+1,2)=ny(i,2)+sys.nx;
    end
end

ny(sys.Np+2,1)=ny(sys.Np+1,1)+size(sys.Ft,1);
ny(sys.Np+2,2)=ny(sys.Np+1,2);

Y.y0=zeros(ny(sys.Np+2,1)+ny(sys.Np+2,2),1);
Y.y1=zeros(ny(sys.Np+2,1)+ny(sys.Np+2,2),1);
prm_fes.hard_const=zeros(ny(sys.Np+2,1),1);
prm_fes.soft_const=zeros(ny(sys.Np+1,2),1);

dual_grad=prm_fes;
dual_variable.hard_const=zeros(ny(sys.Np+2,1),1);
dual_variable.soft_const=zeros(ny(sys.Np+1,2),1);

epsilon_prm=1;

g_nodes.hard_const=zeros(ny(sys.Np+2,1),1);
g_nodes.soft_const=zeros(ny(sys.Np+1,2),1);

for i=1:sys.Np
    if(i==1)
        g_nodes.hard_const(ny(i,1)+1:ny(i+1,1),1)=sys.g{i};
    else
        g_nodes.hard_const(ny(i,1)+1:ny(i+1,1),1)=sys.g{i}(1:ny(i+1,1)-ny(i,1),1);
        g_nodes.soft_const(ny(i,2)+1:ny(i+1,2),1)=sys.g{i}(ny(i+1,1)-ny(i,1)+1:end,1);
    end
end

g_nodes.hard_const(ny(sys.Np+1,1)+1:ny(sys.Np+2,1),1)=sys.gt;

theta=[1 1]';
tic
j=1;
details.term_crit=zeros(1,4);
%%
while(j<ops.steps)
    % step 1: accelerated step
    W.y=Y.y1+theta(2)*(1/theta(1)-1)*(Y.y1-Y.y0);
    
    % step 2: argmin of the lagrangian using dynamic programming
    [Z,Q]=GPAD_dynamic_calculation(sys,Ptree,W,ops.x0);
    
    % details.Z{j}=Z;
    % step 3: proximal step [y_p y_q]'
    % calculate the primal infeasibility conditions.
    % proximal update of y_p is projection on positive quadrant;
    Y.y0=Y.y1;
    for i=1:sys.Np+1
        if(i==sys.Np+1)
            prm_fes.hard_const(ny(i,1)+1:ny(i+1,1),1)=sys.Ft*Z.X(:,i);
            dual_grad.hard_const(ny(i,1)+1:ny(i+1,1),1)=prm_fes.hard_const(ny(i,1)+1:ny(i+1,1),1)...
                -g_nodes.hard_const(ny(i,1)+1:ny(i+1,1),1);
            dual_variable.hard_const(ny(i,1)+1:ny(i+1,1),1)=W.y(ny(i,1)+ny(i,2)+1:ny(i+1,1)+ny(i,2),1);
            Y.y1(ny(i,1)+ny(i,2)+1:ny(i+1,1)+ny(i,2),1)=max(0,W.y(ny(i,1)+ny(i,2)+1:ny(i+1,1)+ny(i,2),1)...
                +ops.alpha*dual_grad.hard_const(ny(i,1)+1:ny(i+1,1),1));
        else
            if(i==1)
                prm_fes.hard_const(ny(i,1)+1:ny(i+1,1),1)=sys.F{i}*Z.X(:,i)+sys.G{i}*Z.U(:,i);
                
                dual_grad.hard_const(ny(i,1)+1:ny(i+1,1),1)=...
                    prm_fes.hard_const(ny(i,1)+1:ny(i+1,1),1)-g_nodes.hard_const(ny(i,1)+1:ny(i+1,1),1);
                Y.y1(ny(i,1)+1:ny(i+1,1),1)=max(0,W.y(ny(i,1)+1:ny(i+1,1),1)+...
                    ops.alpha*dual_grad.hard_const(ny(i,1)+1:ny(i+1,1),1));
                dual_variable.hard_const(ny(i,1)+1:ny(i+1,1),1)=W.y(ny(i,1)+1:ny(i+1,1),1);
            else
                prm_fes_temp=sys.F{i}*Z.X(:,i)+sys.G{i}*Z.U(:,i);
                prm_fes.hard_const(ny(i,1)+1:ny(i+1,1),1)=prm_fes_temp(1:end-sys.nx,1);
                prm_fes.soft_const(ny(i,2)+1:ny(i+1,2),1)=prm_fes_temp(end-sys.nx+1:end,1);
                
                dual_grad.hard_const(ny(i,1)+1:ny(i+1,1),1)=prm_fes_temp(1:end-sys.nx,1)...
                    -g_nodes.hard_const(ny(i,1)+1:ny(i+1,1),1);
                dual_grad.soft_const(ny(i,2)+1:ny(i+1,2),1)=prm_fes_temp(end-sys.nx+1:end,1)...
                    -g_nodes.soft_const(ny(i,2)+1:ny(i+1,2),1);
                
                Y.y1(ny(i,1)+ny(i,2)+1:ny(i+1,1)+ny(i,2),1)=max(0,W.y(ny(i,1)+ny(i,2)+1:ny(i+1,1)+ny(i,2),1)+...
                    ops.alpha*dual_grad.hard_const(ny(i,1)+1:ny(i+1,1),1));
                dual_variable.hard_const(ny(i,1)+1:ny(i+1,1),1)=W.y(ny(i,1)+ny(i,2)+1:ny(i+1,1)+ny(i,2),1);
                dual_variable.soft_const(ny(i,2)+1:ny(i+1,2),1)=W.y(ny(i+1,1)+ny(i,2)+1:ny(i+1,1)+ny(i+1,2),1);
            end
        end
    end
    
    yq=zeros(sys.nx*(sys.Np-1),1);
    for i=2:sys.Np
        yq_temp=W.y(ny(i+1,1)+ny(i,2)+1:ny(i+1,1)+ny(i+1,2),1)/ops.alpha+...
            prm_fes.soft_const(ny(i,2)+1:ny(i+1,2),1);
        for kk=1:sys.nx
            if(yq_temp(kk)<(sys.xs(kk)))
                yq((i-1)*sys.nx+kk,1)=(ops.alpha*yq_temp(kk)+2*V.Qs(kk,kk)*sys.xs(kk,1))/(ops.alpha+2*V.Qs(kk,kk));
            else
                yq((i-1)*sys.nx+kk,1)=yq_temp(kk);
            end
        end
        Y.y1(ny(i+1,1)+ny(i,2)+1:ny(i+1,1)+ny(i+1,2),1)=W.y(ny(i+1,1)+ny(i,2)+1:ny(i+1,1)+ny(i+1,2),1)+...
            ops.alpha*(prm_fes.soft_const(ny(i,2)+1:ny(i+1,2),1)-yq((i-1)*sys.nx+1:i*sys.nx,1));
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
    
    %if epsilon_prm_avg<=ops.primal_inf %average_primal feasibility less
        %details.term_crit(1,2)=1;
        %details.iterate=j;
        %j=10*ops.steps;
    %else
        details.eq_feasibility(iter)=max(abs(prm_fes.soft_const-yq(sys.nx+1:end,1)));
        if(max(abs(prm_fes.soft_const-yq(sys.nx+1:end,1)))<ops.eq_feasibility)
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
                        for i=1:sys.Np
                            prm_cst=prm_cst+Z.U(:,i)'*V.R*Z.U(:,i);
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
                    for i=sys.Np
                        prm_cst=prm_cst+Z.U(:,i)'*V.R*Z.U(:,i);
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

soft_cost=zeros(sys.nx*(sys.Np),1);
for i=2:sys.Np
    soft_cost((i-1)*sys.nx+1:i*sys.nx,1)=prm_fes.soft_const(ny(i,2)+1:ny(i+1,2),1)-yq((i-1)*sys.nx+1:i*sys.nx,1);
end
details.soft_cost=soft_cost;
%details.epsilon_prm_avg= epsilon_prm_avg;
%details.epsilon_prm=epsilon_prm;
end


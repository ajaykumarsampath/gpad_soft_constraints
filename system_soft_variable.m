function [sys_epsilon, V_epsilon] = system_soft_variable( sys,V )
% This function transfor with soft cost as a decision variable. 

%cost function of the soft constrainted system. 
V_epsilon.Q=V.Q;
if(max(eig(V.Qs))>0)
    V_epsilon.R(1:sys.nu,1:sys.nu)=V.R;
    V_epsilon.R(sys.nu+1:sys.nx+sys.nu,sys.nu+1:sys.nx+sys.nu)=V.Qs;
else
    V_epsilon.R=V.R;
end

V_epsilon.Vf=V.Vf;

% system description
sys_epsilon=sys;
if(max(eig(V.Qs))>0)
    sys_epsilon.nu=sys.nx+sys.nu;
    sys_epsilon.B(:,sys.nu+1:sys.nu+sys.nx)=zeros(sys.nx,sys.nx);
else
    sys_epsilon.nu=sys.nu;
end



ny=size(sys.F,1);
F_temp=sys.F;
G_temp=sys.G;
g_temp=sys.g;

sys_epsilon.F=cell(sys.Np,1);
sys_epsilon.G=cell(sys.Np,1);
sys_epsilon.g=cell(sys.Np,1);

if(max(eig(V.Qs))>0)
    for i=1:sys.Np
        sys_epsilon.F{i,1}=[F_temp;-sys.A;zeros(sys.nx)];
        sys_epsilon.G{i,1}(:,1:sys_epsilon.nu)=[G_temp zeros(ny,sys.nx)];
        sys_epsilon.G{i,1}(ny+1:ny+2*sys.nx,:)=[-sys.B -eye(sys.nx);zeros(sys.nx,sys.nu) -eye(sys.nx)];
        sys_epsilon.g{i,1}=[g_temp;-sys.xs;zeros(sys.nx,1)];
    end
else
    for i=1:sys.Np
        sys_epsilon.F{i,1}=F_temp;
        sys_epsilon.G{i,1}=G_temp;
        sys_epsilon.g{i,1}=g_temp;
    end 
end


end


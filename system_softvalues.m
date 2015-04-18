function [sys_dst] = system_softvalues( sys,option )
% This function transfor with soft cost as a decision variable. 

% system description
sys_dst.A=sys.A;
sys_dst.B=sys.B;
sys_dst.nx=sys.nx;
sys_dst.nu=sys.nu;
sys_dst.Np=sys.Np;

sys_dst.F=[sys.A;zeros(2*sys.nu,sys.nx)];
sys_dst.G=[sys.B;sys.G(end-2*sys.nu+1:end,:)];
sys_dst.g=[zeros(sys.nx,1);sys.g(end-2*sys.nu+1:end,1)];

xmax_temp=sys.g(1:sys.nx);
xmin_temp=-sys.g(sys.nx+1:2*sys.nx);

F_temp=sys_dst.F;
G_temp=sys_dst.G;
g_temp=sys_dst.g;

sys_dst.F=cell(sys.Np,1);
sys_dst.G=cell(sys.Np,1);
sys_dst.g=cell(sys.Np,1);
sys_dst.xmin=kron(ones(sys.Np,1),xmin_temp);
sys_dst.xmax=kron(ones(sys.Np,1),xmax_temp);

for i=1:sys.Np
    sys_dst.F{i,1}=F_temp;
    sys_dst.G{i,1}=G_temp;
    sys_dst.g{i,1}=g_temp;
end
sys_dst.gamma_min=option.gamma_min;
sys_dst.gamma_max=option.gamma_max;

end




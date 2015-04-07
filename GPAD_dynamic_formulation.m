function [Ptree] = GPAD_dynamic_formulation( sys,V)
% This function calculate the off-line elements for the dynamic programming
% step of the GPAD algorithm


Ptree=struct('P',cell(1,1),'d',cell(1,1),'f',cell(1,1),'Phi',cell(1,1),'Theta',cell(1,1));


Ptree.P{sys.Np+1,1}=V.Vf;

for i=sys.Np:-1:1
    
    Rbar=2*(V.R+sys.B'*Ptree.P{i+1}*sys.B);
    
    Rbar_inv=Rbar\eye(sys.nu);
    
    Ptree.K{i}=-2*Rbar_inv*(sys.B'*Ptree.P{i+1}*sys.A);%K_{k-1}\
    
    Ptree.Phi{i}=-Rbar_inv*sys.G{i}';%\Phi_{k-1}
    if(i==sys.Np)
        Ptree.Theta{i}=-Rbar_inv*sys.B'*sys.Ft';%\Theta_{k-1}
    else
        Ptree.Theta{i}=-Rbar_inv*sys.B';%\Theta_{k-1}
    end
    
    %terms in the linear cost
    Ptree.d{i}=sys.F{i}+sys.G{i}*Ptree.K{i};%d_{k-1}
    if(i==sys.Np)
        Ptree.f{i}=sys.Ft*(sys.A+sys.B*Ptree.K{i});%f_{k-1}^{(i)}
    else
        Ptree.f{i}=(sys.A+sys.B*Ptree.K{i});%f_{k-1}^{(i)}
    end
    
    %Quadratic cost
    if(i==sys.Np)
        Ptree.P{i}=V.Q+Ptree.K{i}'*V.R*Ptree.K{i}...
            +(sys.A+sys.B*Ptree.K{i})'*Ptree.P{i+1}*(sys.A+sys.B*Ptree.K{i});
    else
        Ptree.P{i}=V.Q+Ptree.K{i}'*V.R*Ptree.K{i}+Ptree.f{i}'*Ptree.P{i+1}*Ptree.f{i};
    end
end

end



function [ Z,Q] = GPAD_dynamic_calculation( sys,Ptree,Y,xinit)
%This function calculate the solution of the dynamic programming step on 
%the tree. All the off-line terms are calculated and passed as input to the
%function. The inital x, and dual varaibles (y) and terminal F_N are
%passed as inputs to this function
% Z is the output and containing 

Z.X=zeros(sys.nx,sys.Np+1);
Z.U=zeros(sys.nu,sys.Np);
S=zeros(sys.nu,sys.Np);

ny=zeros(sys.Np+1,1);
for i=1:sys.Np
    if(i==1)
        ny(i)=size(sys.F{i},1);
    else
        ny(i)=ny(i-1)+size(sys.F{i},1);
    end 
end

ny(sys.Np+1)=ny(sys.Np)+size(sys.Ft,1);

q=zeros(sys.nx,sys.Np);
qt=Y.y(ny(sys.Np)+1:ny(sys.Np+1),1);

for i=sys.Np:-1:1
    if(i==sys.Np)
        S(:,i)=Ptree.Phi{i}*Y.y(ny(i-1)+1:ny(i),1)+Ptree.Theta{i}*qt;
        q(:,i)=Ptree.d{i}'*Y.y(ny(i-1)+1:ny(i),1)+Ptree.f{i}'*qt;
    else
        if(i==1)
            S(:,i)=Ptree.Phi{i}*Y.y(1:ny(i),1)+Ptree.Theta{i}*q(:,i+1);
            q(:,i)=Ptree.d{i}'*Y.y(1:ny(i),1)+Ptree.f{i}'*q(:,i+1);
        else
            S(:,i)=Ptree.Phi{i}*Y.y(ny(i-1)+1:ny(i),1)+Ptree.Theta{i}*q(:,i+1);
            q(:,i)=Ptree.d{i}'*Y.y(ny(i-1)+1:ny(i),1)+Ptree.f{i}'*q(:,i+1);
        end
    end
end
Z.X(:,1)=xinit;
for i=1:sys.Np
    Z.U(:,i)=Ptree.K{i}*Z.X(:,i)+S(:,i);
    Z.X(:,i+1)=sys.A*Z.X(:,i)+sys.B*Z.U(:,i);
end
Q.q=q;
Q.qt=qt;
end


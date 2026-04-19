function W=CSPfeature_s(xTrain,yTrain, nfilter)
%%  train CSP filters


nChannel=size(xTrain,1);
cs=unique(yTrain);
xTrain0=xTrain(:,:,yTrain==cs(1));
xTrain1=xTrain(:,:,yTrain==cs(2));
Sigma0=zeros(nChannel);Sigma1=zeros(nChannel);
for i=1:size(xTrain0,3)
    tmp0=cov(xTrain0(:,:,i)');
    Sigma0=Sigma0+tmp0;
end
for i=1:size(xTrain1,3)
    tmp1=cov(xTrain1(:,:,i)');
    Sigma1=Sigma1+tmp1;
end
Sigma0=Sigma0/size(xTrain0,3);
Sigma1=Sigma1/size(xTrain1,3);
[d,v]=eig((Sigma0+Sigma1)\Sigma0);
% [d,v]=eig(inv(Sigma0)*Sigma1);
[~,v_index]=sort(diag(v),'descend');
d_sort=d(:,v_index);
w0=d_sort(:,1:nfilter); %CSP filters
w1=d_sort(:,end-nfilter+1:end); %CSP filters
W=[w0,w1];


clc
clear
All = [];
ALL_s = [];
tau_selected = 4; % this was determined based on 10-fold cross-validation on the training set

%% Initialization
tau = tau_selected;
if tau == 0
    K = 1;  
else
    K = 2;
end
%% Training stage: run SBLEST on the training set
disp(['FIR filter order: ', num2str(K),  '      Time delay: ', num2str(tau)]);
disp('Running SBLEST : update W, Psi and lambda');

frontal = [3,4,27,28,31]; %F7,F3,F4,F8,Fz
central = [5,8,9,22,23,26,32]; %FC1,FC2,C3,Cz,C4,CP1,CP2
parietal = [11,12,13,19,20];%P3,P4,Pz,P7,P8
occipital = [14,15,16,17,18]; %O1,Oz,O2,PO3,PO4 

left_hemi = [1,2,3,4,5,6,7,8,9,10,11,12,14,15]; % FP1,AF3,F7,F3,FC5,FC1,T7,C3,CP5,CP1,P7,P3,PO3,O1
right_hemi = [17,18,19,20,21,22,23,24,25,26,27,28,29,30]; % FP2,AF4,F4,F8,FC2,FC6,T8,C4,CP2,CP6,P4,P8,PO4,O2


all_acc = [];
for k = 3:8
    x_All = [];
    y_All = [];
    nTrials=[];
	for m =['A','B','C','D']
        str1 =  sprintf("./S%d/S%d_%s.mat",k,k,m);
        load(str1);
        y=y';
        y(find(y==2))=-1;
        x_All = cat(3,x_All, x);
        y_All = cat(1,y_All, y);
        nTrials = [nTrials, size(x,3)];
	end
	for n = 1:4
        idx=sum(nTrials(1:n-1));
        ids=idx+1:idx+nTrials(n);
        X_train=x_All(:,:,:); Y_train=y_All;
        X_test=x_All(:,:,ids); Y_test=y_All(ids);
        X_train(:,:,ids)=[]; Y_train(ids)=[];
        %% Train stage
        [W, alpha, V, Wh] = SBLEST(X_train, Y_train, K, tau);
        %% Test stage : predicte labels in the test set
        R_test = Enhanced_cov_test(X_test, K, tau, Wh);
        predict_Y = R_test*W(:);
        accuracy = compute_acc (predict_Y, Y_test);
        acc(n) = accuracy;
        disp(['Test   Accuracy: ', num2str(accuracy)]);
	end
	all_acc = cat(1,all_acc, acc);
end



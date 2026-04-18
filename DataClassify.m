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


for k = 1:8
    str1 =  sprintf("./S%d/S%d_D.mat",k,k);
    load(str1);
    y=y';
    % y(find(y==2))=-1;
    cv = 10;
    trial = size(y,1);
    num = floor(trial / cv);
    for m = 1:cv
        X_train = x(:,:,:);
        Y_train = y;
        X_test = x(:,:,1+(m-1)*num:m*num);
        Y_test = y(1+(m-1)*num:m*num);
        X_train(:,:,1+(m-1)*num:m*num)=[];
        Y_train(1+(m-1)*num:m*num)=[];

        [Xs, Xt] = CSPfeature_s(X_train, Y_train, X_test, 5);
        SVM = fitcsvm(Xs, Y_train);
        predict_Y = predict(SVM,Xt);
        % %% Train stage
        % [W, alpha, V, Wh] = SBLEST(X_train, Y_train, K, tau);
        % %% Test stage : predicte labels in the test set
        % R_test = Enhanced_c3.ov_test(X_test, K, tau, Wh);
        % predict_Y = R_test*W(:);
        accuracy = compute_acc (predict_Y, Y_test);
        acc(m) = accuracy;
        disp(['Test   Accuracy: ', num2str(accuracy)]);
    end
    All = [All,mean(acc)];
    ALL_s = [ALL_s, sqrt(var(acc))];
end

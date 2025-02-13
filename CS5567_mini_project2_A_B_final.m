% Load the bodyfat dataset %
[X,T] = bodyfat_dataset;

% Part A %%%%
fprintf('Part A:\n')

% Find the number of inputs and outputs %
[numInputs, numSamples] = size(X);
[numOutputs, ~] = size(T);

% Compute the correlation %
for i = 1:numInputs
    corrCoeff = corrcoef(X(i,:), T);
    fprintf('Correlation coefficient of input %d with the output: %.4f\n', i, corrCoeff(1,2));
end
% Output: 
% the inputs absolute correlation coefficients are listed from highest to
% lowest
% 
% Input 6: 0.8134
% Input 5: 0.7026
% Input 7: 0.6252
% Input 2: 0.6124
% Input 8: 0.5596
% Input 9: 0.5087
% Input 11: 0.4933
% Input 4: 0.4906
% Input 12: 0.3614
% Input 13: 0.3466
% Input 1: 0.2915
% Input 10: 0.2660
% Input 3: 0.0895
% The top four inputs have correlation coefficients greater than 0.6 in absolute value, 
% indicating a strong linear relationship with the output variable. 

% A regression model using the top four inputs listed above %
selectedInputs = [X(6,:); X(5,:); X(7,:); X(2,:)];

% Fit a linear regression model using the selected inputs %
mdl = fitlm(selectedInputs', T);

% Display the model coefficients %
disp(mdl.Coefficients);

% Display the model summary
disp(mdl);

% Output:
% Coefficients:
% (Intercept)       -45.681      
%     x1            0.99111   
%     x2            -0.0027546    
%     x3            -0.0029795     
%     x4            -0.14713    
%
% Number of observations: 252, Error degrees of freedom: 247
% Root Mean Squared Error: 4.47
% R-squared: 0.719,  Adjusted R-Squared: 0.714
% F-statistic vs. constant model: 158, p-value = 8.07e-67

% Split the data into training (50%) and test (50%) sets %
numTrain = floor(numSamples/2);
numTest = numSamples - numTrain;
trainInputs = selectedInputs(:, 1:numTrain)';
trainTargets = T(1:numTrain);
testInputs = selectedInputs(:, numTrain+1:end)';
testTargets = T(numTrain+1:end);

% Train a linear regression model on the training set %
mdl2 = fitlm(trainInputs, trainTargets);

% Predict the output variable for the test set %
predTargets = predict(mdl2, testInputs);

% Compute the mean squared error %
mse = 0;
for i = 1:numTest
    mse = mse + (predTargets(i) - testTargets(i)).^2;
end
mse = mse / numTest;
% Display the mean squared error %
disp(['Average MSE From Test Data: ' num2str(mse)]);

% Output:
% Average MSE From Test Data: 21.403
 
% Part B %%%%
% Task 1 %%%%
fprintf('\nPart B\n')
fprintf('Task 1:\n')
% Set up neural network (10-node one hidden layer MLP) %
net = fitnet(10); 
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0;
net.performFcn = 'mse';

% Train and evaluate the network 10 times %
trainMSEs = zeros(10, 1);
valMSEs = zeros(10, 1);
task1nets = {};
for i = 1:10
    % Reset network weights %
    net = init(net);
    % Add net to list %
    task1nets{end+1} = net;
    % Train network %
    [net, tr] = train(net, X, T);

    % Evaluate on training set %
    Ytrain = net(X(:, tr.trainInd));
    Ttrain = T(:, tr.trainInd);

    trainMSE = 0;
    numTrain = length(Ytrain);
    for j = 1:numTrain
        trainMSE = trainMSE + (Ttrain(j) - Ytrain(j)).^2;
    end
    trainMSE = trainMSE / numTrain;
    trainMSEs(i) = trainMSE;

    % Evaluate on validation set %
    Yval = net(X(:, tr.valInd));
    Tval = T(:, tr.valInd);

    valMSE = 0;
    numVal = length(Yval);
    for j = 1:numVal
        valMSE = valMSE + (Tval(j) - Yval(j)).^2;
    end
    valMSE = valMSE / numVal;
    valMSEs(i) = valMSE;
    
end

% Report mean and variance of MSEs
disp('1. Simple MLP with 10 nodes, 80-20-0 training-validation-test partitioning ratios:');
disp(['Mean training MSE: ' num2str(mean(trainMSEs)) ', variance: ' num2str(var(trainMSEs))]);
disp(['Mean validation MSE: ' num2str(mean(valMSEs)) ', variance: ' num2str(var(valMSEs))]);

% Output
% 1. Simple MLP with 10 nodes, 80-20-0 training-validation-test partitioning ratios:
% Mean training MSE: 16.0611, variance: 31.5062
% Mean validation MSE: 27.3888, variance: 59.3156

% Part B %%%%
% Task 2 %%%%
fprintf('\nPart B\n')
fprintf('Task 2:\n')
% Set up neural network (2-node one hidden layer MLP) %
net = fitnet(2);
net.divideParam.trainRatio = 0.3;
net.divideParam.valRatio = 0.7;
net.divideParam.testRatio = 0;
net.performFcn = 'mse';

% Train and evaluate the network 10 times with 2 nodes hidden layer size%
trainMSEs_2 = zeros(10, 1);
valMSEs_2 = zeros(10, 1);
task2nets2node = {};
for i = 1:10
    % Reset network weights %
    net = init(net);

    % Train network %
    [net, tr] = train(net, X, T);
    % Add net to list %
    task2nets2node{end+1} = net;
    % Evaluate on training set %
    Ytrain = net(X(:, tr.trainInd));
    Ttrain = T(:, tr.trainInd);

    trainMSE_2 = 0;
    numTrain = length(Ytrain);
    for j = 1:numTrain
        trainMSE_2 = trainMSE_2 + (Ttrain(j) - Ytrain(j)).^2;
    end
    trainMSE_2 = trainMSE_2 / numTrain;
    trainMSEs_2(i) = trainMSE_2;

    % Evaluate on validation set %
    Yval = net(X(:, tr.valInd));
    Tval = T(:, tr.valInd);

    valMSE_2 = 0;
    numVal = length(Yval);
    for j = 1:numVal
        valMSE_2 = valMSE_2 + (Tval(j) - Yval(j)).^2;
    end
    valMSE_2 = valMSE_2 / numVal;
    valMSEs_2(i) = valMSE_2;
    
end

% Report mean and variance of MSEs
disp('2. MLP with 2 nodes in hidden layer, 30-70-0 training-validation-test partitioning ratios:');
disp(['Mean training MSE: ' num2str(mean(trainMSEs_2)) ', variance: ' num2str(var(trainMSEs_2))]);
disp(['Mean validation MSE: ' num2str(mean(valMSEs_2)) ', variance: ' num2str(var(valMSEs_2))]);

% Set up neural network (50-node one hidden layer MLP) %
net = fitnet(50);
net.divideParam.trainRatio = 0.3;
net.divideParam.valRatio = 0.7;
net.divideParam.testRatio = 0;
net.performFcn = 'mse';

% Train and evaluate the network 10 times with 50 nodes hidden layer size%
trainMSEs_50 = zeros(10, 1);
valMSEs_50 = zeros(10, 1);
task2nets50node = {};
for i = 1:10
    % Reset network weights %
    net = init(net);

    % Train network %
    [net, tr] = train(net, X, T);

    % Add net to list %
    task2nets50node{end+1} = net;

    % Evaluate on training set %
    Ytrain = net(X(:, tr.trainInd));
    Ttrain = T(:, tr.trainInd);

    trainMSE_50 = 0;
    numTrain = length(Ytrain);
    for j = 1:numTrain
        trainMSE_50 = trainMSE_50 + (Ttrain(j) - Ytrain(j)).^2;
    end
    trainMSE_50 = trainMSE_50 / numTrain;
    trainMSEs_50(i) = trainMSE_50;

    % Evaluate on validation set %
    Yval = net(X(:, tr.valInd));
    Tval = T(:, tr.valInd);

    valMSE_50 = 0;
    numVal = length(Yval);
    for j = 1:numVal
        valMSE_50 = valMSE_50 + (Tval(j) - Yval(j)).^2;
    end
    valMSE_50 = valMSE_50 / numVal;
    valMSEs_50(i) = valMSE_50;
    
end

% Report mean and variance of MSEs
disp('MLP with 50 nodes in hidden layer, 30-70-0 training-validation-test partitioning ratios:');
disp(['Mean training MSE: ' num2str(mean(trainMSEs_50)) ', variance: ' num2str(var(trainMSEs_50))]);
disp(['Mean validation MSE: ' num2str(mean(valMSEs_50)) ', variance: ' num2str(var(valMSEs_50))]);

% Output:
% Part B
% Task 2:
% 2. MLP with 2 nodes in hidden layer, 30-70-0 training-validation-test partitioning ratios:
% Mean training MSE: 15.0667, variance: 3.256
% Mean validation MSE: 26.5229, variance: 66.3409
% MLP with 50 nodes in hidden layer, 30-70-0 training-validation-test partitioning ratios:
% Mean training MSE: 5.2321, variance: 40.8902
% Mean validation MSE: 98.7919, variance: 612.7226

% Part B %%%%
% Task 3 %%%%
fprintf('\nPart B\n')
fprintf('Task 3:\n')

% Train and evaluate the network 10 times with 50 nodes hidden layer size and regularization set at 0.1%
trainMSEs_50_01 = zeros(10, 1);
valMSEs_50_01 = zeros(10, 1);
task3nets_1percent = {};

for i = 1:10
    % Reset network weights %
    net = init(net);

    % Set regularization parameter to 0.1%
    net.trainParam.weightDecay = 0.1;

    % Train network %
    [net, tr] = train(net, X, T);

    % Add net to list %
    task3nets_1percent{end+1} = net;
    % Evaluate on training set %
    Ytrain = net(X(:, tr.trainInd));
    Ttrain = T(:, tr.trainInd);

    trainMSE_50_01 = 0;
    numTrain = length(Ytrain);
    for j = 1:numTrain
        trainMSE_50_01 = trainMSE_50_01 + (Ttrain(j) - Ytrain(j)).^2;
    end
    trainMSE_50_01 = trainMSE_50_01 / numTrain;
    trainMSEs_50_01(i) = trainMSE_50_01;

    % Evaluate on validation set %
    Yval = net(X(:, tr.valInd));
    Tval = T(:, tr.valInd);

    valMSE_50_01 = 0;
    numVal = length(Yval);
    for j = 1:numVal
        valMSE_50_01 = valMSE_50_01 + (Tval(j) - Yval(j)).^2;
    end
    valMSE_50_01 = valMSE_50_01 / numVal;
    valMSEs_50_01(i) = valMSE_50_01;
    
end

% Report mean and variance of MSEs for regularization set at 0.1%
disp('3. MLP with 50 nodes in hidden layer and regularization set at 0.1, 30-70-0 training-validation-test partitioning ratios:');
disp(['Mean training MSE: ' num2str(mean(trainMSEs_50_01)) ', variance: ' num2str(var(trainMSEs_50_01))]);
disp(['Mean validation MSE: ' num2str(mean(valMSEs_50_01)) ', variance: ' num2str(var(valMSEs_50_01))]);

% Train and evaluate the network 10 times with 50 nodes hidden layer size and regularization set at 0.5%
trainMSEs_50_005 = zeros(10, 1);
valMSEs_50_005 = zeros(10, 1);
task3nets_5percent = {};
for i = 1:10
    % Reset network weights %
    net = init(net);

    % Set regularization parameter to 0.5%
    net.trainParam.weightDecay = 0.005;

    % Train network %
    [net, tr] = train(net, X, T);

    % Add net to list %
    task3nets_5percent{end+1} = net;

    % Evaluate on training set %
    Ytrain = net(X(:, tr.trainInd));
    Ttrain = T(:, tr.trainInd);

    trainMSE_50_005 = 0;
    numTrain = length(Ytrain);
    for j = 1:numTrain
        trainMSE_50_005 = trainMSE_50_005 + (Ttrain(j) - Ytrain(j)).^2;
    end
    trainMSE_50_005 = trainMSE_50_005 / numTrain;
    trainMSEs_50_005(i) = trainMSE_50_005;

    % Evaluate on validation set %
    Yval = net(X(:, tr.valInd));
    Tval = T(:, tr.valInd);

    valMSE_50_005 = 0;
    numVal = length(Yval);
    for j = 1:numVal
        valMSE_50_005 = valMSE_50_005 + (Tval(j) - Yval(j)).^2;
    end
    valMSE_50_005 = valMSE_50_005 / numVal;
    valMSEs_50_005(i) = valMSE_50_005;
    
end

% Report mean and variance of MSEs for regularization set at 0.5%
disp('3. MLP with 50 nodes in hidden layer and regularization set at 0.5%, 30-70-0 training-validation-test partitioning ratios:');
disp(['Mean training MSE: ' num2str(mean(trainMSEs_50_005)) ', variance: ' num2str(var(trainMSEs_50_005))]);
disp(['Mean validation MSE: ' num2str(mean(valMSEs_50_005)) ', variance: ' num2str(var(valMSEs_50_005))]);
% Output:
% 3. MLP with 50 nodes in hidden layer and regularization set at 0.1, 30-70-0 training-validation-test partitioning ratios:
% Mean training MSE: 16.2568, variance: 1155.988
% Mean validation MSE: 95.2001, variance: 272.4407
% 3. MLP with 50 nodes in hidden layer and regularization set at 0.5%, 30-70-0 training-validation-test partitioning ratios:
% Mean training MSE: 9.8743, variance: 163.5429
% Mean validation MSE: 129.7403, variance: 1940.4029
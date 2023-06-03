% Clear the workspace
clear
close all
clc

% Import the data into a table (which is converted to an array)
PPG_data = table2array(readtable("Data_Collection_Testing.xlsx"));

% Selecting only the rows that have data assigned to them
PPG_data = PPG_data(1:128,:);

% Normalisation of features
for a = 2:12
    X_bar = sum(PPG_data(:,a))/length(PPG_data(:,a));
    sigmasum = 0;
    for b = 1:length(PPG_data(:,1))
        sigmasum = sigmasum + (PPG_data(b,a)-X_bar)^2;
    end
    sigma = sqrt(sigmasum/2);

    for c = 1:length(PPG_data(:,1))
        PPG_data(c,a) = (PPG_data(c,a)-X_bar)/sigma;
    end
end

% Initial shuffling of the array
array_length = length(PPG_data(:,1));
n = randperm(array_length);
PPG_data = PPG_data(n,:);

% Convert all BGL categories into one-hot encoding
for i = 1:array_length
    if PPG_data(i,13) == 1 % Low BGL
        PPG_data(i,13:15) = [0.8, 0.2, 0.2];
    elseif PPG_data(i,13) == 2 % Normal BGL
        PPG_data(i,13:15) = [0.2, 0.8, 0.2];
    elseif PPG_data(i,13) == 3 % High BGL
        PPG_data(i,13:15) = [0.2, 0.2, 0.8];
    end
end

% Split data into 70% for training and 30% for testing
SeventyPercent = round(0.7*array_length, 0);
PPGDataTrain = PPG_data(1:SeventyPercent,:);
PPGDataTest = PPG_data(SeventyPercent+1:array_length,:);

% Training data (x=input, d=correct output)
x = PPGDataTrain(:,2:12)';
d = PPGDataTrain(:,13:15)';

% Testing data
x_test = PPGDataTest(:,2:12)';
d_test = PPGDataTest(:,13:15)';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Neural Network Parameters %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Design parameters
eta = 0.007; % Learning parameter
n_epoch = 5000; % Number of epochs
nodes_input_L = 11; % 11 nodes for 11 features
nodes_first_HL = 5;
nodes_second_HL = 3;
nodes_output_L = 3; % 3 nodes for 3 different categories

% Set randomised initial weights
w1 = randi([1 9], nodes_first_HL,(nodes_input_L+1))./100;
w2 = randi([1 9], nodes_second_HL,(nodes_first_HL+1))./100;
w3 = randi([1 9], nodes_output_L,(nodes_second_HL+1))./100;

% Set dimensions of all layers (i.e. number of nodes of each layer)
% initially set to 1 so that we can skip over the bias
x1 = ones(1,(nodes_input_L+1));
x2 = ones(1, (nodes_first_HL+1));
x3 = ones(1, (nodes_second_HL+1));
y_out = ones(1, nodes_output_L);

% Empty accuracy arrays to be used for Accuracy vs Epoch graphs
accuracy1 = zeros(1,n_epoch);
accuracy2 = zeros(1,n_epoch);

% Empty arrays to be used for Weights vs Epoch
weight1 = zeros(1,n_epoch);
weight2 = zeros(1,n_epoch);
weight3 = zeros(1,n_epoch);


%%%%%%%%%%%%
% Training %
%%%%%%%%%%%%


% Outer loop - epochs
for epoch = 1:n_epoch

    % Shuffling of training set with each epoch
    training_array_length = length(x(1,:)); % Number of data instances in training array
    [sxR, sxC] = size(x); % Size of training features
    [sdR, sdC] = size(d); % Size of training labels
    to_be_shuffled = zeros(13, sxC); % Create empty array
    to_be_shuffled(1:sxR, 1:sxC) = x; % Transfer 'x' to empty array
    to_be_shuffled(sxR+1:14, 1:sdC) = d; % Transfer 'd' labels to corresponding features
    % Shuffle the new array
    n = randperm(training_array_length);
    to_be_shuffled = to_be_shuffled(:,n);
    % Split the new array back to 'x' and 'd'. The reason x and d were
    % initially joined together as one and then shuffled is because
    % shuffling them separately would not guarantee that the labels
    % correspond to their given features.
    x = to_be_shuffled(1:11, 1:training_array_length);
    d = to_be_shuffled(12:14, 1:training_array_length);

    % This same process is then repeated for the testing data
    testing_array_length = length(x_test(1,:)); % Number of data instances in training array
    [sxR, sxC] = size(x_test); % Size of training features
    [sdR, sdC] = size(d_test); % Size of training labels
    to_be_shuffled = zeros(13, sxC); % Create empty array
    to_be_shuffled(1:sxR, 1:sxC) = x_test; % Transfer 'x_test' to empty array
    to_be_shuffled(sxR+1:14, 1:sdC) = d_test; % Transfer 'd_test' labels to corresponding features
    % Shuffle the new array
    n = randperm(testing_array_length);
    to_be_shuffled = to_be_shuffled(:,n);
    x_test = to_be_shuffled(1:11, 1:testing_array_length);
    d_test = to_be_shuffled(12:14, 1:testing_array_length);

    % Inner loop - iterates through all training data
    for i = 1:training_array_length
    
        %%%%%%%%%%%%%%%%
        % Forward pass %
        %%%%%%%%%%%%%%%%

        % Input layer
        for k = 2:1:(nodes_input_L+1)
            x1(k) = x((k-1),i);
        end

        % First hidden layer
        for k = 2:1:(nodes_first_HL+1)
            v = sum((x1).*(w1(k-1,:)));
            x2(k) = ReLU(v); % Activation function for first hidden layer
        end

        % Second hidden layer
        for k = 2:1:(nodes_second_HL+1)
            v = sum((x2).*(w2((k-1),:)));
            x3(k) = purelin(v); % Activation function for second hidden layer
        end

        % Output layer
        for k = 1:nodes_output_L
            v = sum((x3).*(w3(k,:)));
            y_out(k) = logsig(v); % Activation function for output layer
        end
    

        %%%%%%%%%%%%%%%%%%%
        % Backpropagation %
        %%%%%%%%%%%%%%%%%%%
    

        % Calculate deltas for output layer
        deltas3 = zeros(1, nodes_output_L);
        for k = 1:nodes_output_L
            deltas3(k) = (d(k,i)-y_out(k))*(1-y_out(k))*y_out(k);
        end
        
        % Update weights for output layer
        for k = 1:(nodes_second_HL+1)
            for m = 1:nodes_output_L
                w3(m,k) = w3(m,k) + eta * deltas3(m) * x3(k);
            end
        end
        
        
        % Calculate deltas for second hidden layer
        deltas2 = zeros(1, nodes_second_HL);
        for k = 1:nodes_second_HL
            total = 0;
            for m = 1:nodes_output_L
                total = total + deltas3(m)*w3(m,k+1);
            end
            deltas2(k) = total;
        end
        
        % Update weights for second hidden layer
        for k = 1:(nodes_first_HL+1)
            for m = 1:nodes_second_HL
                w2(m,k) = w2(m,k) + eta * deltas2(m) * x2(k);
            end
        end


        % Calculate deltas for first hidden layer
        deltas1 = zeros(1, nodes_first_HL);
        for k = 1:nodes_first_HL
            if x2(k+1) == 0
                derivative = 0;
            else
                derivative = 1;
            end
            total = 0;
            for m = 1:nodes_second_HL
                total = total + deltas2(m)*w2(m,k+1);
            end
            deltas1(k) = total * derivative;
        end

        % Update weights for first hidden layer
        for k = 1:(nodes_input_L+1)
            for m = 1:nodes_first_HL
                w1(m,k) = w1(m,k) + eta * deltas1(m) * x1(k);
            end
        end
    
    end

    % Recording the value of the current (random) weight from each layer
    weight1(epoch) = w1(2,2);
    weight2(epoch) = w2(1,3);
    weight3(epoch) = w3(3,4);
    

    % Number of correctly tested cases (to be incremented in loop)
    correct1 = 0;
    correct2 = 0;

    % Number of testing instances
    array_length_testing = length(x_test(1,:));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Testing on Training Data %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for i = 1:training_array_length

        % Forward pass
        for k = 2:1:(nodes_input_L+1)
            x1(k) = x((k-1),i);
        end
    
        for k = 2:1:(nodes_first_HL+1)
            v = sum((x1).*(w1(k-1,:)));
            x2(k) = ReLU(v);
        end
    
        for k = 2:1:(nodes_second_HL+1)
            v = sum((x2).*(w2((k-1),:)));
            x3(k) = purelin(v);
        end
    
        for k = 1:nodes_output_L
            v = sum((x3).*(w3(k,:)));
            y_out(k) = logsig(v);
        end


        % If loop to convert one-hot encoded output to string
        if y_out(1)>0.6 && y_out(2)<0.4 && y_out(3)<0.4
            thiscase = "Low";
        elseif y_out(1)<0.4 && y_out(2)>0.6 && y_out(3)<0.4
            thiscase = "Normal";
        elseif y_out(1)<0.4 && y_out(2)<0.4 && y_out(3)>0.6
            thiscase = "High";
        else
            thiscase = "Invalid";
        end

        % If loop to convert known correct values to string
        if d(:,i) == [0.8; 0.2; 0.2]
            truecase = "Low";
        elseif d(:,i) == [0.2; 0.8; 0.2]
            truecase = "Normal";
        elseif d(:,i) == [0.2; 0.2; 0.8]
            truecase = "High";
        else
            truecase = "Invalid";
        end

    
        % Increment correctly tested cases if testing is accurate
        if thiscase==truecase
            correct1 = correct1 + 1;
        end

    end

    % Returns accuracy of NN (correctly tested training cases over total training cases)
    accuracy1(epoch) = (correct1/training_array_length) * 100;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Testing on Testing Data %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%

    for i = 1:array_length_testing

        % Forward pass
        for k = 2:1:(nodes_input_L+1)
            x1(k) = x_test((k-1),i);
        end
    
        for k = 2:1:(nodes_first_HL+1)
            v = sum((x1).*(w1(k-1,:)));
            x2(k) = ReLU(v);
        end
    
        for k = 2:1:(nodes_second_HL+1)
            v = sum((x2).*(w2((k-1),:)));
            x3(k) = purelin(v);
        end
    
        for k = 1:nodes_output_L
            v = sum((x3).*(w3(k,:)));
            y_out(k) = logsig(v);
        end


        % If loop to convert one-hot encoded output to string
        if y_out(1)>0.6 && y_out(2)<0.4 && y_out(3)<0.4
            thiscase = "Low";
        elseif y_out(1)<0.4 && y_out(2)>0.6 && y_out(3)<0.4
            thiscase = "Normal";
        elseif y_out(1)<0.4 && y_out(2)<0.4 && y_out(3)>0.6
            thiscase = "High";
        else
            thiscase = "Invalid";
        end

        % If loop to convert known correct values to string
        if d_test(:,i) == [0.8; 0.2; 0.2]
            truecase = "Low";
        elseif d_test(:,i) == [0.2; 0.8; 0.2]
            truecase = "Normal";
        elseif d_test(:,i) == [0.2; 0.2; 0.8]
            truecase = "High";
        else
            truecase = "Invalid";
        end

    
        % Increment correctly tested cases if testing is accurate
        if thiscase==truecase
            correct2 = correct2 + 1;
        end

    end

    % Returns accuracy of NN (correctly tested cases over total tested cases)
    accuracy2(epoch) = (correct2/array_length_testing) * 100;

end

% Plotting the Accuracy vs Epoch graph
axis = 1:n_epoch;
plot(axis,accuracy1)
title("Accuracy against epochs for 3-category NN", "FontSize", 17)
xlabel("Epoch", "FontSize",16)
ylabel("Accuracy (%)", "FontSize",16)
grid on
hold on
plot(axis,accuracy2)
ylim([0 100])
legend("Training Data", "Testing Data", "Location","west", "FontSize", 16)

% Plotting the Weights vs Epoch graph
figure
plot(axis, weight1)
hold on
plot(axis, weight2)
plot(axis, weight3)
title("Weights against Epochs", "FontSize", 17)
xlabel("Epoch", "FontSize",16)
ylabel("Value of Weights", "FontSize",16)
legend("First Hidden Layer", "Second Hidden Layer", "Output Layer", "Location","northwest", "FontSize", 16)
grid on


% ReLU Function
function fi = ReLU(x)
    if x<0
        fi = 0;
    else
        fi = x;
    end
end

% Settings
val_size = 5000;
m = 100;

% batch size | eta min | eta max | n_s | number of cycles
% For first method
GDparams = [100, 0, 2e-2, 81, 1];

% batch size | eta min | eta max | n_s | number of epochs
% For second method
GDparams = [100, 1e-5, 1e-1, 500, 20];
seed = 20;
plotting = "True";

% ========================================================================

% Load data
[trainX, trainY, trainy, valX, valY, valy, testX, testY, testy]  = LoadAll(val_size);
[trainX, trainY, trainy, valX, valY, valy, testX, testY, testy]  = LoadSingle();

% Get sizes
[d, n] = size(trainX);
[K, ~] = size(trainY);

% Initialize the parameters
[W, b] = InitializeParameters(K, m, d, seed);

% Parameters for gradient descent
lambda = 1e-03;

% Run gradient decent
[eta_min, eta_max] = MiniBatchGDEta2(trainX, trainY, trainy, valX, valY, valy, GDparams, W, b, lambda);

% % batch size | eta min | eta max | n_s | number of cycles
GDparams = [100, eta_min, eta_max, 500, 10];

% Run gradient decent
[Wstar, bstar] = MiniBatchGD(trainX, trainY, trainy, valX, valY, valy, GDparams, W, b, lambda, plotting);

% Calculate accuracies
TrainingAccuracy = ComputeAccuracy(trainX, trainy, Wstar, bstar)
TestAccuracy = ComputeAccuracy(testX, testy, Wstar, bstar)


% ========================================================================


function [P, H] = evaluateClassifier(X, W, b)
    H = max(0, W{1} * X + b{1});
    s = W{2} * H + b{2};
    P = exp(s) ./ sum(exp(s));
end

function [J, l_cross] = ComputeCost(X, Y, W, b, lambda)
    penalizer = 0;
    for i=1:2
        penalizer = penalizer + lambda * sum(W{i}.^2, 'all');
    end
    l_cross = ComputeLoss(X, Y, W, b);
    [~, n] = size(X);
    J = l_cross / n + penalizer;
end

function L = ComputeLoss(X, Y, W, b)
    P = evaluateClassifier(X, W, b);
    L = -sum(log(sum(P .* Y)));
end

function acc = ComputeAccuracy(X, y, W, b)
    [~, k] = max(evaluateClassifier(X, W, b));
    acc = sum(k == y) / length(y);
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, H, lambda)
    % Step 1
    G = -(Y-P);
    [~, n] = size(X);
    
    % Step 2: Layer 2
    L_W2 = G * H.' / n;
    grad_b2 = G * ones([n, 1]) / n;
    
    % Step 3: Propagate to layer 1
    G = W{2}.' * G;
    H(H>0) = 1; % Indicator function
    G = G .* H;
    
    % Step 4: Layer 1
    L_W1 = G * X.' / n;
    grad_b1 = G * ones([n, 1]) / n;
    
    % Step 5: Regularization
    grad_W1 = L_W1 + 2 * lambda * W{1};
    grad_W2 = L_W2 + 2 * lambda * W{2};
    
    % Put in cells
    grad_W = {grad_W1, grad_W2};
    grad_b = {grad_b1, grad_b2};
end

function [relative_error] = ComputeRelativeError(analyticalX, numericalX)
    relative_error = abs(analyticalX - numericalX) ./ ...
        max(eps, abs(analyticalX) + abs(numericalX));
%     abs_error = abs(analyticalX - numericalX);
end

function [Wstar, bstar] = MiniBatchGD(X, Y, y, valX, valY, valy, GDparams, W, b, lambda, plotting)
    % ORIGINAL GD

    % Extract parameters
    n_batch = GDparams(1);
    eta_min = GDparams(2);
    eta_max = GDparams(3);
    n_s = GDparams(4);
    n_cycles = GDparams(5);
    [~, n] = size(X);
    n_epochs = n_s / (n / n_batch) * n_cycles * 2;
    
    % Costs, losses and accuracies
    J_train = [];
    L_train = [];
    J_val = [];
    L_val = [];
    Acc_train = [];
    Acc_val = [];
    
    % Counting the steps
    t = 0;
    
    % Epoch-loop
    for i=1:n_epochs
        % Batch-loop
        
        % Deterministic order
        batch_order = 1:n/n_batch;
        for j=batch_order
            % Extract a batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            X_batch = X(:, j_start:j_end);
            Y_batch = Y(:, j_start:j_end);

            % Compute P
            [P, H] = evaluateClassifier(X_batch, W, b);

            % Compute gradients
            [grad_W, grad_b] = ComputeGradients(X_batch, Y_batch, P, W, H, lambda);
            
            % Update eta
            l = floor(t / (2 * n_s));
            if 2 * l * n_s <= t && t <= (2 * l + 1) * n_s
                eta = eta_min + (t - 2 * l * n_s) / n_s * (eta_max - eta_min);
            else % (2 * l + 1) * n_s <= t <= 2 * (l + 1) * n_s
                eta = eta_max - (t - (2 * l + 1) * n_s) / n_s * (eta_max - eta_min);
            end
            
            % Update parameters
            for k=1:2
                W{k} = W{k} - eta * grad_W{k};
                b{k} = b{k} - eta * grad_b{k};
            end
            
            t = t + 1;
        end
        
        if plotting == "True"
            % Save the current cost
            [J, L] = ComputeCost(valX, valY, W, b, lambda);
            J_val = [J_val, J];
            L_val = [L_val, L / length(valy)];

            [J, L] = ComputeCost(X, Y, W, b, lambda);
            J_train = [J_train, J];
            L_train = [L_train, L / length(y)];

            Acc_train = [Acc_train, ComputeAccuracy(X, y, W, b)];
            Acc_val = [Acc_val, ComputeAccuracy(valX, valy, W, b)];

            if mod(100 * i / n_epochs, 10) == 0
                fprintf('%0.8f %% \n', floor(100 * i / n_epochs));
            end
        end
    end
    if plotting == "True"
        % Plot the cost function
        subplot(1,3,1)
        plot(J_train, 'b')
        hold all
        plot(J_val, 'r')
        legend('training set', 'validation set')
        xlabel('epoch')
        ylabel('cost')

        % Plot the total loss
        subplot(1,3,2)
        plot(L_train, 'b')
        hold all
        plot(L_val, 'r')
        legend('training set', 'validation set')
        xlabel('epoch')
        ylabel('total loss')

        % Plot the accuracies
        subplot(1,3,3)
        plot(Acc_train, 'b')
        hold all
        plot(Acc_val, 'r')
        legend('training set', 'validation set')
        xlabel('epoch')
        ylabel('Accuracy')

        "MiniBatchGD complete"
    end

    % Save optimal values
    Wstar = W;
    bstar = b;    
end

function [Wstar, bstar] = MiniBatchGD2(X, Y, y, valX, valY, valy, GDparams, W, b, lambda, plotting)
    % PLOTS MORE OFTEN COMPARED TO ORIGINAL
    
    % Extract parameters
    n_batch = GDparams(1);
    eta_min = GDparams(2);
    eta_max = GDparams(3);
    n_s = GDparams(4);
    n_cycles = GDparams(5)
    [~, n] = size(X);
    n_epochs = n_s / (n / n_batch) * n_cycles * 2
    
    % Costs, losses and accuracies
    J_train = [];
    L_train = [];
    J_val = [];
    L_val = [];
    Acc_train = [];
    Acc_val = [];
    step = [];
    
    % Counting the steps
    t = 0;
    
    % Epoch-loop
    for i=1:floor(n_epochs)
        % Batch-loop
        
        % Deterministic order
        n_batches = n/n_batch;
        batch_order = 1:n_batches;
        for j=batch_order
            % Extract a batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            X_batch = X(:, j_start:j_end);
            Y_batch = Y(:, j_start:j_end);

            % Compute P
            [P, H] = evaluateClassifier(X_batch, W, b);

            % Compute gradients
            [grad_W, grad_b] = ComputeGradients(X_batch, Y_batch, P, W, H, lambda);
            
            % Update eta
            l = floor(t / (2 * n_s));
            if 2 * l * n_s <= t && t <= (2 * l + 1) * n_s
                eta = eta_min + (t - 2 * l * n_s) / n_s * (eta_max - eta_min);
            else % (2 * l + 1) * n_s <= t <= 2 * (l + 1) * n_s
                eta = eta_max - (t - (2 * l + 1) * n_s) / n_s * (eta_max - eta_min);
            end
            
            % Update parameters
            for k=1:2
                W{k} = W{k} - eta * grad_W{k};
                b{k} = b{k} - eta * grad_b{k};
            end
            
            if mod(j, n_batches / 10) == 0 && plotting == "True"
                step = [step, t];
                
                [J, L] = ComputeCost(valX, valY, W, b, lambda);
                J_val = [J_val, J];
                L_val = [L_val, L / length(valy)];

                [J, L] = ComputeCost(X, Y, W, b, lambda);
                J_train = [J_train, J];
                L_train = [L_train, L / length(y)];

                Acc_train = [Acc_train, ComputeAccuracy(X, y, W, b)];
                Acc_val = [Acc_val, ComputeAccuracy(valX, valy, W, b)];

                if mod(100 * i / n_epochs, 10) == 0
                    fprintf('%0.8f %% \n', floor(100 * i / n_epochs));
                end
            end
            
            t = t + 1;
        end

    end
    if plotting == "True"
        % Plot the cost function
        subplot(1,3,1)
        plot(step, J_train, 'color', [0, 0.5, 0])
        hold all
        plot(step, J_val, 'r')
        legend('training', 'validation')
        xlabel('update step')
        ylabel('cost')
        title('Cost plot')
        yl = ylim;
        ylim([0, yl(2)]);

        % Plot the total loss
        subplot(1,3,2)
        plot(step, L_train, 'color', [0, 0.5, 0])
        hold all
        plot(step, L_val, 'r')
        legend('training', 'validation')
        xlabel('update step')
        ylabel('total loss')
        title('Total loss plot')
        yl = ylim;
        ylim([0, yl(2)]);

        % Plot the accuracies
        subplot(1,3,3)
        plot(step, Acc_train, 'color', [0, 0.5, 0])
        hold all
        plot(step, Acc_val, 'r')
        legend('training', 'validation')
        xlabel('update step')
        ylabel('accuracy')
        title('Accuracy plot')
        yl = ylim;
        ylim([0, yl(2)]);

        "MiniBatchGD complete"
    end

    % Save optimal values
    Wstar = W;
    bstar = b;    
end

function [eta_min, eta_max] = MiniBatchGDEta2(X, Y, y, valX, valY, valy, GDparams, W, b, lambda)
    % METHOD 2
    
    % Extract parameters
    n_batch = GDparams(1);
    eta_min = GDparams(2);
    eta_max = GDparams(3);
    n_s = GDparams(4);
    n_epochs = GDparams(5)
    [~, n] = size(X);
%     n_epochs = floor(n_s / (n / n_batch) * n_cycles * 2)
    
    % Costs, losses and accuracies
    Acc_val = [];
    etas = [];
    
    % Counting the steps
    t = 0;
    
    % Epoch-loop
    for i=1:n_epochs+1
        % Batch-loop
        
        % Update eta
        eta = eta_min + (i-1) / n_epochs * (eta_max - eta_min);
        
        % Deterministic order
        batch_order = 1:n/n_batch;
        for j=batch_order
            % Extract a batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            X_batch = X(:, j_start:j_end);
            Y_batch = Y(:, j_start:j_end);

            % Compute P
            [P, H] = evaluateClassifier(X_batch, W, b);

            % Compute gradients
            [grad_W, grad_b] = ComputeGradients(X_batch, Y_batch, P, W, H, lambda);
                       
            
            % Update parameters
            for k=1:2
                W{k} = W{k} - eta * grad_W{k};
                b{k} = b{k} - eta * grad_b{k};
            end
            
            t = t + 1;
        end

        % Save values
        acc = ComputeAccuracy(valX, valy, W, b);
        Acc_val = [Acc_val, acc];
        etas = [etas, eta];

    end

    plot(etas, Acc_val, 'k')
    hold on
    
    % Find the good learning rates
    slope = diff(Acc_val) ./ diff(etas);
    
    % Eta min
    index_eta_min = find(slope > 0,1); % Sufficient increase, > 0
    eta_min = etas(index_eta_min);
    
    % Eta max
    index_eta_max = find(slope< 0, 1); % Sufficient decrease, < 0
    eta_max = etas(index_eta_max);
    
    % Plot optimal etas
    xline(eta_min, 'r--');
    xline(eta_max, 'r--');
    
    % Labels
    xlabel("Learning rate")
    ylabel("Accuracy")
 
end

function [eta_min, eta_max] = MiniBatchGDEta(X, Y, y, valX, valY, valy, GDparams, W, b, lambda)
    % METHOD 1

    % Extract parameters
    n_batch = GDparams(1);
    eta_min = GDparams(2);
    eta_max = GDparams(3);
    n_s = GDparams(4);
    
    % Costs, losses and accuracies
    Acc_val = [];
    etas = [];
    
    % Counting the steps
    t = 0;
    
    % Deterministic order
    batch_order = 1:min(n_s, length(y) / n_batch);
    for j=batch_order
        % Extract a batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        X_batch = X(:, j_start:j_end);
        Y_batch = Y(:, j_start:j_end);

        % Compute P
        [P, H] = evaluateClassifier(X_batch, W, b);

        % Compute gradients
        [grad_W, grad_b] = ComputeGradients(X_batch, Y_batch, P, W, H, lambda);

        % Update eta
        l = floor(t / (2 * n_s));
        if 2 * l * n_s <= t && t <= (2 * l + 1) * n_s
            eta = eta_min + (t - 2 * l * n_s) / n_s * (eta_max - eta_min);
        else % (2 * l + 1) * n_s <= t <= 2 * (l + 1) * n_s
            eta = eta_max - (t - (2 * l + 1) * n_s) / n_s * (eta_max - eta_min);
        end          

        % Save values
        acc = ComputeAccuracy(valX, valy, W, b);
        Acc_val = [Acc_val, acc];
        etas = [etas, eta];

        % Update parameters
        for k=1:2
            W{k} = W{k} - eta * grad_W{k};
            b{k} = b{k} - eta * grad_b{k};
        end

        t = t + 1;
    end

    plot(etas, Acc_val, 'k')
    hold on
    
    % Find the good learning rates
    slope = diff(Acc_val) ./ diff(etas);
    
    % Eta min
    index_eta_min = find(slope > 1,1); % Sufficient increase, > 1
    eta_min = max(0.0001, etas(index_eta_min));
    
    % Eta max
    index_eta_max = find(slope<-3, 1); % Sufficient decrease, < -3
    eta_max = etas(index_eta_max);
    
    % Plot optimal etas
    xline(eta_min, 'r--');
    xline(eta_max, 'r--');
    
    % Labels
    xlabel("Learning rate")
    ylabel("Accuracy")
end

function [X, Y, y] = LoadBatch(filename)
    % Loads batch data
    % X - pixel data d x n
    % y - label data 1 x n
    % Y - one-hot encoded label data K x n
    addpath Datasets/cifar-10-batches-mat/;
    A = load(filename);
    X = double(A.data).';
    y = (A.labels+1).';
    Y = (y==(1:10).');
end

function [trainX, valX, testX] = normalizeData(trainX, valX, testX)
    %NORMALIZE 
    mean_X = mean(trainX, 2);
    std_X = std(trainX, 0, 2);
    
    trainX = normalize(trainX, mean_X, std_X);
    valX = normalize(valX, mean_X, std_X);
    testX = normalize(testX, mean_X, std_X);

end

function normalizedX = normalize(X, mean_X, std_X)
    X = X - repmat(mean_X, [1, size(X, 2)]);
    normalizedX = X ./ repmat(std_X, [1, size(X, 2)]);
end

function [trainX, trainY, trainy, valX, valY, valy, testX, testY, testy]  = LoadSingle()
    % Load data
    [trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
    [valX, valY, valy] = LoadBatch('data_batch_2.mat');
    [testX, testY, testy] = LoadBatch('test_batch.mat');

    % Pre-process data
    [trainX, valX, testX] = normalizeData(trainX, valX, testX);
    
    "Loading complete"
end

function [trainX, trainY, trainy, valX, valY, valy, testX, testY, testy]  = LoadAll(val_size)
    % Load all data
    [trainX1, trainY1, trainy1] = LoadBatch('data_batch_1.mat');
    [trainX2, trainY2, trainy2] = LoadBatch('data_batch_2.mat');
    [trainX3, trainY3, trainy3] = LoadBatch('data_batch_3.mat');
    [trainX4, trainY4, trainy4] = LoadBatch('data_batch_4.mat');
    [trainX5, trainY5, trainy5] = LoadBatch('data_batch_5.mat');
    [testX, testY, testy] = LoadBatch('test_batch.mat');

    trainX = [trainX1, trainX2, trainX3, trainX4, trainX5];
    trainY = [trainY1, trainY2, trainY3, trainY4, trainY5];
    trainy = [trainy1, trainy2, trainy3, trainy4, trainy5];

    valX = trainX(:, 1:val_size);
    valY = trainY(:, 1:val_size);
    valy = trainy(1:val_size);

    trainX = trainX(:, val_size+1:end);
    trainY = trainY(:, val_size+1:end);
    trainy = trainy(val_size+1:end);

    % Pre-process data
    [trainX, valX, testX] = normalizeData(trainX, valX, testX);
    
    "Loading complete"
end

function [W, b] = InitializeParameters(K, m, d, seed)
    rng(seed)

    W1 = randn([m, d])/sqrt(d);
    b1 = zeros([m, 1]);
    
    W2 = randn([K, m])/sqrt(m);
    b2 = zeros([K, 1]);
    
    W = {W1, W2};
    b = {b1, b2};
end

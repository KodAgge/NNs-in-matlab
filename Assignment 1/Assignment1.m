% Load data
[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
[valX, valY, valy] = LoadBatch('data_batch_2.mat');
[testX, testY, testy] = LoadBatch('test_batch.mat');

% Pre-process data
trainX = normalize(trainX);
valX = normalize(valX);
testX = normalize(testX);

% Get sizes
[d, n] = size(trainX);
[K, ~] = size(trainY);

% Initialize the parameters
W = randn([K, d])*0.01;
b = randn([K, 1])*0.01;

% Parameters for gradient descent
lambda = 1;

% batch size | learning rate | number of epochs
GDparams = [100, 0.001, 40];

% Run gradient decent
[Wstar, bstar] = MiniBatchGD(trainX, trainY, valX, valY, GDparams, W, b, lambda);

% Calculate accuracies
TrainingAccuracy = ComputeAccuracy(trainX, trainy, Wstar, bstar)
TestAccuracy = ComputeAccuracy(testX, testy, Wstar, bstar)

% Visualize the weight matrix
VisualizeW(Wstar)

% ========================================================================
% Check if gradient is correct

% Dimensionality reductions for checking gradients
batch_size = 1000;
dimensions = 200;
trainX = trainX(1:dimensions, 1:batch_size);
trainY = trainY(:, 1:batch_size);
trainy = trainy(:, 1:batch_size);
W = W(:, 1:dimensions);

% Computing gradients
P = evaluateClassifier(trainX, W, b);
[grad_W, grad_b] = ComputeGradients(trainX, trainY, P, W, lambda);
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(trainX, trainY, ...
    W, b, lambda, 1e-6);
[ngrad_b2, ngrad_W2] = ComputeGradsNum(trainX, trainY, ...
    W, b, lambda, 1e-6);

% Errors as percentage of total
Slowb = DifferenceGradients(grad_b, ngrad_b, 1e-6)
SlowW = DifferenceGradients(grad_W, ngrad_W, 1e-6)
Fastb = DifferenceGradients(grad_b, ngrad_b2, 1e-6)
FastW = DifferenceGradients(grad_W, ngrad_W2, 1e-6)

% ========================================================================

function P = evaluateClassifier(X, W, b)
    s = W * X + b;
    P = exp(s) ./ sum(exp(s));
end

function [J, l_cross] = ComputeCost(X, Y, W, b, lambda)
    penalizer = lambda * sum(W.^2, 'all');
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

function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
    G = -(Y-P);
    [~, n] = size(X);
    L_W = G * X.' / n;
    grad_b = G * ones([n, 1]) / n;
    grad_W = L_W + 2 * lambda * W;
end

function [accuracy] = DifferenceGradients(analyticalG, numericalG, tol)
    relative_error = ComputeRelativeError(analyticalG, numericalG);
    errors = relative_error(relative_error > tol);
    accuracy = 100 * numel(errors) / numel(relative_error);
end

function [relative_error] = ComputeRelativeError(analyticalX, numericalX)
    relative_error = abs(analyticalX - numericalX) ./ ...
        max(eps, abs(analyticalX) + abs(numericalX));
%     abs_error = abs(analyticalX - numericalX);
end

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

    no = size(W, 1);
    d = size(X, 1);

    grad_W = zeros(size(W));
    grad_b = zeros(no, 1);

    for i=1:length(b)
        b_try = b;
        b_try(i) = b_try(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        b_try = b;
        b_try(i) = b_try(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b(i) = (c2-c1) / (2*h);
    end

    for i=1:numel(W)

        W_try = W;
        W_try(i) = W_try(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);

        W_try = W;
        W_try(i) = W_try(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);

        grad_W(i) = (c2-c1) / (2*h);
    end

end

function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

    no = size(W, 1);
    d = size(X, 1);

    grad_W = zeros(size(W));
    grad_b = zeros(no, 1);

    c = ComputeCost(X, Y, W, b, lambda);

    for i=1:length(b)
        b_try = b;
        b_try(i) = b_try(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b(i) = (c2-c) / h;
    end

    for i=1:numel(W)   

        W_try = W;
        W_try(i) = W_try(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);

        grad_W(i) = (c2-c) / h;
    end

end

function [Wstar, bstar] = MiniBatchGD(X, Y, valX, valY, GDparams, W, b, lambda)
    % Extract parameters
    n_batch = GDparams(1);
    eta = GDparams(2);
    n_epochs = GDparams(3);
    [~, n] = size(X);
    
    % Costs and losses
    J_train = [];
    L_train = [];
    J_val = [];
    L_val = [];
    
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
            P = evaluateClassifier(X_batch, W, b);

            % Compute gradients
            [grad_W, grad_b] = ComputeGradients(X_batch, Y_batch, P, W, lambda);

            % Update parameters
            W = W - eta * grad_W;
            b = b - eta * grad_b;
        end
        % Save the current cost
        [J, L] = ComputeCost(valX, valY, W, b, lambda);
        J_val = [J_val, J];
        L_val = [L_val, L];
        
        [J, L] = ComputeCost(X, Y, W, b, lambda);
        J_train = [J_train, J];
        L_train = [L_train, L];

    end
    
    % Plot the cost function
    subplot(1,2,1)
    plot(J_train, 'g')
    hold all
    plot(J_val, 'r')
    legend('training set', 'validation set')
    xlabel('epoch')
    ylabel('cost')
    
    % Plot the total loss
    subplot(1,2,2)
    plot(L_train, 'g')
    hold all
    plot(L_val, 'r')
    legend('training set', 'validation set')
    xlabel('epoch')
    ylabel('total loss')

    % Save optimal values
    Wstar = W;
    bstar = b;
end

function VisualizeW(W)
    for i=1:10
        im = reshape(W(i, :), 32, 32, 3);
        s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
        s_im{i} = permute(s_im{i}, [2, 1, 3]);
    end
    figure(2);
    montage(s_im)
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

function normalizedX = normalize(X)
%NORMALIZE 

mean_X = mean(X, 2);
std_X = std(X, 0, 2);


X = X - repmat(mean_X, [1, size(X, 2)]);
normalizedX = X ./ repmat(std_X, [1, size(X, 2)]);
end
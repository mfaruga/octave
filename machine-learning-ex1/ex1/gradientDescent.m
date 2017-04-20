function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    %prediction = theta'*X';    
    %prediction = 0;
    %for sample = 1:m,
      % TODO remove hard-coding and use vector with any number of fields
    %  prediction = prediction + theta(1) * X(sample,1) + theta(2) * X(sample,2);            
    %end;

    %tempTheta1 =  theta(1) - alpha / m * (prediction-y(sample)) * X(sample,1);
    %tempTheta2 =  theta(2) - alpha / m * (prediction-y(sample)) * X(sample,2);  
    
    % correct vectorized
    % theta = theta - (alpha/m * ((X * theta) - y)' * X)';    

    first = 0.0;
    second = 0.0;
    for sample = 1:m,      
      temp = X(sample,1) * theta(1) + X(sample,2) * theta(2) - y(sample);  
      first = first + temp * X(sample,1);
      second = second + temp * X(sample,2);
    end;
    
    first = first * alpha/m;
    second = second * alpha/m;    
    
    theta(1) = theta(1) - first;
    theta(2) = theta(2) - second;
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

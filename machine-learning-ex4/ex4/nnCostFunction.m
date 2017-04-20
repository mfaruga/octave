function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%


X = [ones(m, 1) X];

vectorized = 0;
if vectorized 
  % Vectorized implementation of forward propagation
  A2 = sigmoid(X*Theta1');
  A2 = [ones(size(A2,1),1), A2];
  A3 = sigmoid(A2*Theta2');

  % non-vectorized implementation
  for sampleIndex = 1:m
    % convert to vector with 0's and 1 for proper classification
    expected = (1:max(y) == y(sampleIndex));
    % cut out predictions
    predictions = A3(sampleIndex, :);
    % calculate cost and add cost value to total cost
    cost = log(predictions)*(-1*expected)' - log(1-predictions)*(1-expected)';
    J = J + 1/m*sum(cost);    
  end  
end


% non vectorized implementation of forward propagation

for sampleIndex = 1:m

  a1 = X(sampleIndex,:);
  z2 = a1*Theta1';
  a2 = sigmoid(z2);
  a2 = [1, a2];
  z3 = a2*Theta2';
  a3 = sigmoid(z3);
  
  expected = (1:max(y) == y(sampleIndex));
  predictions = a3;  
  cost = log(predictions)*(-1*expected)' - log(1-predictions)*(1-expected)';
  J = J + cost;
  
  delta3 = predictions - expected;
  delta2 = (delta3*Theta2)(2:end) .* sigmoidGradient(z2);
    
  Theta1_grad = Theta1_grad + delta2' * a1;
  Theta2_grad = Theta2_grad + delta3' * a2;
end
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;
J = J * 1/m;

Reg1 = Theta1 * (lambda/m);
Reg1(:,1) = 0;
Theta1_grad = Theta1_grad + Reg1;

Reg2 = Theta2 * (lambda/m);
Reg2(:,1) = 0;
Theta2_grad = Theta2_grad + Reg2;


% TODO vectorized implemenentation
%EXPECTED = (1:10 == y);
%cost = log(A3)*(-1*EXPECTED)' - log(1-A3)*(1-EXPECTED)';

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% non-vectorized implementation
%for sampleIndex = 1:m
    
  % convert to vector with 0's and 1 for proper classification
%  expected = (1:10 == y(sampleIndex));
  % cut out predictions
%  predictions = A3(sampleIndex, :);

  % delta 3 is just predictions - expected values
%  Delta3 = predictions - expected;
  
  % delta 2 
  
%end

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

 
J = J + (lambda/(2*m)) * ( sum(sum(Theta2(:,2:end).^2)) + sum(sum(Theta1(:,2:end).^2)));


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

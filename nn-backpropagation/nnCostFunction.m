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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Think about how to implement this dynamically, which works for any number of hidden layers, instead of hard coding layers.

% add ones to the X matrix, and initialze a1
a1 = [ones(m,1) X];
% size(a1)

% caculate z2
z2 = a1 * Theta1';
% size(z2)
a2 = sigmoid(z2);

% caculate z3
% add ones to a2
a2 = [ones(m,1) a2];
% size(a2)

z3 = a2 * Theta2';
a3 = sigmoid(z3);
% size(a3)

% initialize values for cost function
J = 0;
Kcost = zeros(size(a3));
Icost = zeros(m,1);
C = 0;

% note: to optimize this block of code?
for k = 1:num_labels
  C1 = (-1 * (y == k)) .* log(a3(:,k));
  C2 = (1 - (y == k)) .* log(1 - a3(:,k));
  Kcost(:,k) = C1 - C2;
end

% the ORDER of the summation cannot be wrong
% sum along rows as the sum of k units
Icost = sum(Kcost,2);
% sum of all examples
C = sum(Icost);
J = (1/m) * C;

% initalize values for regularization
Reg1 = 0;
Reg2 = 0;

% MUST exclude the first column in the Theta Matrices
KR1 = sum(Theta1(:,[2:size(Theta1,2)]) .^ 2, 2);
KR2 = sum(Theta2(:,[2:size(Theta2,2)]) .^ 2, 2);

Reg1 = sum(KR1);
Reg2 = sum(KR2);
Reg = (lambda/(2*m)) * (Reg1 + Reg2);
J = (1/m) * C + Reg;

d_3 = zeros(m,num_labels);

% build y_matrix
y_m = zeros(m,num_labels);
for t = 1:m
  for i = 1:num_labels
    if y(t) == i
      y_m(t,i) = 1;
    end
  end
end
% size(y_m)

% caculate delta_3
d_3 = a3 - y_m;
% size(d_3)
d_2  = d_3*Theta2(:,[2:end]) .* sigmoidGradient(z2);
% size(d_2)
D1 = d_2' * a1;
% size(D1)
D2 = d_3' * a2;
% size(D2)

Theta1_grad = (1/m) * D1;
Theta2_grad = (1/m) * D2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

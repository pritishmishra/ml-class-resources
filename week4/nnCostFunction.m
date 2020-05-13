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

delta1 = Theta1_grad;
delta2 = Theta2_grad;


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

y_mat = zeros(m,num_labels);


% Size of y is 5000 * 1. Each value is 0-9.
% We need to convert it to 5000 * 10 mat.
% each row should look 0's and only 1 for specific value.
for i = 1:m,
  index = y(i,:);
  if index == 0 
    y_mat(i,10) = 1; 
  else  
    y_mat(i,index) = 1;
  endif 
end

for i = 1:m,
 % Dimension of X is 5000 * 400
 % Extracting the 400 pixels for each input
  X_pixels = X(i, :);

 % Dimension of X_pixels is 1 * 400. X_pixels should be a column vector
 % We need to add bias unit of 1 to make X_pixels 1 * 401.
 X_pixels = [1 X_pixels];

 %display(size(X));

 % Dimensions of Theta1 is 26 * 401
 z_2 = Theta1*X_pixels';
 a_2 = sigmoid(z_2);

 %display(size(a_2));

 % Dimension of a_2 is 25 * 1. So, need to add bias.
  a_2_bias = [1; a_2];

 % Dimension of Theta 2 is 10 * 26. So, a_3 becomes 10 * 1 output matrix.
  z_3 = Theta2 * a_2_bias;

  a_3 = sigmoid(z_3);
  
  J += -y_mat(i, :) * log(a_3) - (1-y_mat(i, :)) * log(1-a_3); 

  % ------------------BACKPROPAGATION-------------------
  sigma_3 = a_3 - y_mat(i);
  sigma_2 = (Theta2' * sigma_3 .*sigmoidGradient([1;z_2]));

  delta1 += (sigma_2 * X_pixels)(2:end,:);
  delta2 += sigma_3 * a_2_bias'; 
 end
 
 % ------------COST FUNCTION---------------
 J = J/m;

 % Removing bias
 Theta1_reg = Theta1(:,2:end);
 Theta2_reg = Theta2(:,2:end);

 % sum(sum()) computes sum of all elemets of the matrix
 Theta1_reg_prod = sum(sum(Theta1_reg.^2, 2));
 Theta2_reg_prod = sum(sum(Theta2_reg.^2, 2));
 
 reg_param = (lambda/(2*m)) * (Theta1_reg_prod + Theta2_reg_prod);
 
 % ---------------REGULARISED COST FUNCTION------------
 J += reg_param;


 % ------------------BACKPROPAGATION-------------------
 Theta1_grad = delta1./m;
 Theta2_grad = delta2./m;

 % ------------------BACKPROPAGATION WITH REGULARISATION-------------------
 Theta1_grad += (lambda/m) * [zeros(size(Theta1,1), 1) Theta1_reg];
 Theta2_grad += (lambda/m) * [zeros(size(Theta2,1), 1) Theta2_reg];
 
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

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
  [max_value, max_index] = max(a_3, [], 1);
  p(i, 1) = max_index;
 end


% =========================================================================


end

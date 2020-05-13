function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% h remains same for regularized
h = sigmoid(X*theta);

% We need to introduce another theta term; since in regularised param, theta-0
% should not be used. And, in this code, theta-0 is actually index 1. So, we 
% need to pick from index-2 of theta

theta_shift = theta(2:size(theta));

% We further need to concatenate this with 0

theta_reg = [0;theta_shift];


% Additional params to J in regularized
% formula says theta_reg squared. But, since theta_reg is vector, I think it is 
% represented as below
J = (1/m)*(-y'* log(h) - (1 - y)'* log(1-h)) + (lambda/(2*m))*theta_reg'*theta_reg;

% Following formula works for both cases of gradient, since for theta_reg=0, the lamba term=0

grad = (1/m)*((X'*(h-y)) + (lambda*theta_reg))
 





% =============================================================

end

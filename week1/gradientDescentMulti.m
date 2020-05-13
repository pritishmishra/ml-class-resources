function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %

    h = X*theta;
   
    count = 2:size(theta);

    theta_zero = theta(1) - (alpha * (1/m) * sum(h-y));
    theta_new = [theta_zero];
    %disp(theta_new);   
  
    for i = count, 
      x = X(:,i);
      theta_temp  = theta(i) - (alpha * (1/m) * sum((h - y) .* x));
      theta_new = [theta_new; theta_temp];
      %disp(theta_new);
    end

    theta = theta_new;
 

    % Shorter vectorized version of the above code
    %h = X * theta;
    %theta = theta - (alpha/m) * (X' * (h - y));

    %fprintf("iter:%f, theta:%f\n",iter, theta);
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
    %disp(J_history(iter));
end
    fprintf("Final cost: %f\n",J_history(iter-1));
end

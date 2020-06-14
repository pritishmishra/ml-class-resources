function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;



model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

predictions = svmPredict(model, Xval);
min_error = mean(double(predictions ~= yval));
C_opt = C;
sigma_opt = sigma;
disp(min_error);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_arr = [1, 3, 9, 27, 81];
sigma_arr = [0.03, 0.01, 0.1, 0.3, 0.9, 0.27, 0.81]

for i = 1:length(C_arr),
  for j = i:length(sigma_arr),
    C = C_arr(i);
    sigma = sigma_arr(j);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    error_temp = mean(double(predictions ~= yval));
    disp(error_temp);
    if error_temp < min_error
       disp("Error reduced");
       min_error = error_temp;
       C_opt = C;
       sigma_opt = sigma;
    endif
  end
end

C = C_opt;
sigma = sigma_opt;

% =========================================================================

end

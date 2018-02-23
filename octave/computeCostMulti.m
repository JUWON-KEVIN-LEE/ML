function J = computeCostMulti(X, y, theta)

% default variable
m = length(y); % number of training examples
J = 0;

hypothesis = X * theta;

errors = hypothesis .- y;

squareOfErrors = (errors) .^ 2;

sumOfErrors = sum(squareOfErrors);

J = sumOfErrors / (2 * m);

endfunction
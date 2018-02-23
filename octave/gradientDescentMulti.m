function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

% default variable
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % X : m x n matrix
    % theta : n x 1 vector
    % hypothesis : m x 1 vector
    hypothesis = X * theta;
    
    % errors : m x 1 vector
    errors = hypothesis .- y;
    
    % errors' : 1 x m matrix
    % X : m x n
    % newDecrement : 1 x n matrix
    newDecrement = alpha * (errors' * X) / m;
    
    theta = theta - newDecrement';

    J_history(iter) = computeCostMulti(X, y, theta);
endfor

endfunction
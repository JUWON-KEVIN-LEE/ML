function [X_norm, mu, sigma] = featureNormalize(X)

% default variable
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% number of features x1, x1, ... xn
numberOfFeatures = columns(X_norm);

% 1. z-score standardization : z = X - ¥ì / ¥ò
for 1:numberOfFeatures,
  % mean of ith vector in X
  meanOfXi = mean(X(:,i)); 
  mu(:, i) = meanOfXi;
  
  % X - ¥ì
  X_norm(:, i) = X_norm(:, i) .- mu(:, i);
  
  % X - ¥ì / ¥ò
  standardDeviationOfXi = std(X(:, i));
  sigma(:, i) = standardDeviationOfXi;
  
  X_norm(:, i) = X_norm(:, i) ./ sigma(:, i);
endfor

% 2. Min-Max Scaling : X_norm = X - X_min / X_max - X_min
% more ... ?

% variance : (1/N) * square(X - ¥ì)
% standard deviation : ¥ò, roots(variance) = roots(E(X^2) - (E(X))^2)

endfunction
function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
n=size(X,2);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

prediction=X*theta;
thetas=0;
if length(theta) > 2
  thetas=theta(2:end);
end  
J=(1/(2*m))*sum((prediction-y).^2)+(lambda/(2*m))*sum(thetas.^2);
temp=theta;
temp(1)=0;
for i=1:n
  grad(i)=(1/m).*sum((prediction-y)'*X(:,i))+ (lambda/m).*temp(i);
end








% =========================================================================

grad = grad(:);

end

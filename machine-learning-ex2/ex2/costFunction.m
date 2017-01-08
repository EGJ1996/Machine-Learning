function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

oneVec=ones(1,size(y,1));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%fprintf('Sigmoid of 0 is %f',sigmoid(0));
%fprintf('Sigmoid of a very large number is %f',sigmoid(100000));
predictions=X*theta;
h=sigmoid(predictions);
prod1=(-1*y).*log(h);
prod2=(1-y).*log(1-h);
J=(1/m)*sum(prod1-prod2);

%fprintf('Row size of predictions is %i\n',size(predictions,1));
%fprintf('Column size of predictions is %i\n',size(predictions,2));
%fprintf('Row size of X is %i\n',size(X,1));
%fprintf('Column size of X is %i\n',size(X,2));
for i=1:size(X,2)
  grad(i)=(1/m)*sum((h-y).*X(:,i));
end

%fprintf('Size of gradient is %i and size of theta is %i\n',size(grad,1),size(theta,1));
% =============================================================

end

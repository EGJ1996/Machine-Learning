function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
numFeatures=size(X,2);
fprintf('Row number of X is %i \n',size(X,1));
fprintf('Column number of X is %i \n',size(X,2));
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
  


  

  predictions=X*theta - y; 
   for i=1:numFeatures
     % S(i) = sum(predictions.*featureNormalize(X)(1)(:,1));
     % fprintf('S(i) is %i \n',S(i));
     a=featureNormalize(X)(1);
     disp(a);
   end
   
   % theta=theta-alpha*(1/m)*S';




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end

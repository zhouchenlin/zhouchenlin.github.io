%% ADM method for ellipsoid fitting.
% X is a n by (p+1) matrix, where the last column is 1.

function Q = ADM(X)

n = size(X,1);
p = size(X,2) - 1;
m = (p+1)*(p+2)/2;

tic;
Tpre = tic;
Xhat = zeros(n,m);
for i = 1:n
    Xhat(i,:) = quad_vec(X(i,:));
end

HAT_D = Xhat' * Xhat;
[U,S] = eig(HAT_D);

R = eye(p+1,p+1);
Q = eye(p+1,p+1);
mu = zeros(m,1);
beta = 0.1;
rho = 1.02;
beta_max = 2600;
converge = 0;
iter = 0;
stop = 1e-3;
a = zeros(m,1);
b = zeros(m,1);
bprev = zeros(m,1);
aprev = zeros(m,1);
Mask = zeros(m,1);

indTmp = 1;
for i = 1:p
    Mask(indTmp,1) = 1;
    indTmp = indTmp + i + 1;
end

preSec = toc(Tpre);
fprintf('the pre_processing time is: %f \n', preSec);

while ~converge
    
    iter = iter + 1;
    
    %update a;
    HAT_S = 2*S + beta * eye(m,m);
    inv_S = diag(1./diag(HAT_S));
    g  = -(mu + beta* b);
    a  = U*inv_S*(U'*((Mask + (Mask'*U)*inv_S*(U'*g)*Mask )./((Mask'*U)*inv_S*(U'*Mask)) - g));
    
    %update b;
    R = vec2mat(a - mu/beta, p+1);
    Q = R;
    [V,D] = eig(R(1:p,1:p));
    for i = 1:p
        D(i,i) = max(D(i,i),0);
    end
    Q(1:p,1:p) = V*D*V';
    b = mat2vec(Q);
    
    %update mu
    mu = mu + beta*(b-a);
    
    %update beta
    beta = min([beta*rho, beta_max]);
    
    if max([norm(a - aprev), norm(b - bprev), norm(b - a)]) < stop
        converge = 1;
    end
    
    bprev = b;
    aprev = a;
end

fprintf('The algorithm converges after %d iters \n', iter);
TimeC = toc;
fprintf('Total time for ADM algorithm is : %f \n', TimeC); 

residual_L2 = 0;
residual_L1 = 0;
residual_Linf = 0;
for i = 1:n
    residual_L2 = residual_L2 + (X(i,:) * Q * X(i,:)')^2;
    residual_L1 = residual_L1 + abs(X(i,:) * Q * X(i,:)');
    residual_Linf = max(abs(X(i,:) * Q * X(i,:)'), residual_Linf);
end
residual_L2 = residual_L2 /n;
residual_L1 = residual_L1 /n;
fprintf('The mean L2 residual of ADM is : %f \n',residual_L2);
fprintf('The mean L1 residual of ADM is : %f \n',residual_L1);
fprintf('The Linf residual of ADM is: %f \n',residual_Linf);
disp(Q);

if p == 2
    figure(1);
    hold on
    
    % str = [num2str(b(1)),'*x^2 + ',num2str(b(2)),'*x*y + ',num2str(b(4)),' *y^2 + ', num2str(b(3)),' *x + ',...
    %     num2str(b(5)), ' *y + ' num2str(b(6)), ' = 0'];
    str = [num2str(Q(1,1)),'*x^2 + ',num2str(2*Q(1,2)),'*x*y + ',num2str(Q(2,2)),' *y^2 + ', num2str(2*Q(3,1)),' *x + ',...
        num2str(2*Q(3,2)), ' *y + ' num2str(Q(3,3)), ' = 0'];
    
    h3 = ezplot(str,[-5,5]);
    set(h3,'Color','r');
    legend(h3,'ADMM');
end
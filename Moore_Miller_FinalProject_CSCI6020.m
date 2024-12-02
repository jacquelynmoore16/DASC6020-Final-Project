clc 
clear 

% Load the data
%data = readtable('iphc_fulldataset.csv');
data=readtable('iphc_fulldataset_full_depth.csv'); 

% Get unique regulation areas
unique_areas = unique(data.RegArea);

% Display unique regulation areas and prompt user for selection
disp('Available Regulation Areas:');
disp(unique_areas);

% User input for regulation area
regulation_area = input('Enter the regulation area you want to filter by: ', 's');

% Check if the entered regulation area is valid
if ~ismember(regulation_area, unique_areas)
    error('Invalid regulation area. Please enter a valid area from the list.');
end

% Filter data for the selected regulation area
filtered_data = data(strcmp(data.RegArea, regulation_area), :);

% Check if filtered data is empty
if isempty(filtered_data)
    error('Filtered data is empty. Check the regulation area specified.');
end

% Preprocess the filtered data
filtered_data = rmmissing(filtered_data);
filtered_data.stnno = grp2idx(filtered_data.stnno); % Convert categorical to numeric

% Define features and target
X = filtered_data{:, {'avg_temp', 'avg_oxy', 'avg_chloro', 'avg_salin', 'year'}};
y = filtered_data.annual_landings;

% Check if there are enough samples to split
if height(filtered_data) < 5
    error('Not enough data to split into training and testing sets.');
end

% Split the data
cv = cvpartition(height(filtered_data), 'HoldOut', 0.2);
idx = cv.test;

X_train = X(~idx, :);
y_train = y(~idx);
X_test = X(idx, :);
y_test = y(idx);

% Standardize the features
mu = mean(X_train);
sigma = std(X_train);
X_train_scaled = (X_train - mu) ./ sigma;
X_test_scaled = (X_test - mu) ./ sigma;

% Optimize KNN model using fitcknn for regression
best_k = 1;
best_r_squared = -Inf;
for k = 1:20
    knnModel = fitcknn(X_train_scaled, y_train, 'NumNeighbors', k, 'Distance', 'euclidean', 'Standardize', 1);
    y_pred = predict(knnModel, X_test_scaled);
    r_squared = 1 - (sum((y_test - y_pred).^2) / sum((y_test - mean(y_test)).^2));
    if r_squared > best_r_squared
        best_r_squared = r_squared;
        best_k = k;
    end
end
fprintf('Best k: %d with R-squared: %.4f\n', best_k, best_r_squared);

% Train the best KNN model
knnModel = fitcknn(X_train_scaled, y_train, 'NumNeighbors', best_k, 'Distance', 'euclidean', 'Standardize', 1);

% Custom cross-validation
K = 5; % Number of folds
cv_indices = crossvalind('Kfold', y_train, K);
cv_r_squared = zeros(K, 1);

for i = 1:K
    test_idx = (cv_indices == i);
    train_idx = ~test_idx;
    
    X_train_cv = X_train_scaled(train_idx, :);
    y_train_cv = y_train(train_idx);
    X_test_cv = X_train_scaled(test_idx, :);
    y_test_cv = y_train(test_idx);
    
    knnModel_cv = fitcknn(X_train_cv, y_train_cv, 'NumNeighbors', best_k, 'Distance', 'euclidean', 'Standardize', 1);
    y_pred_cv = predict(knnModel_cv, X_test_cv);
    
    cv_r_squared(i) = 1 - (sum((y_test_cv - y_pred_cv).^2) / sum((y_test_cv - mean(y_test_cv)).^2));
end

mean_cv_r_squared = mean(cv_r_squared);
fprintf('Cross-validated R-squared: %.4f\n', mean_cv_r_squared);

% Evaluate the model
mse = mean((y_test - y_pred).^2);
mae = mean(abs(y_test - y_pred));
r_squared = 1 - (sum((y_test - y_pred).^2) / sum((y_test - mean(y_test)).^2));

% Display the results
fprintf('Mean Squared Error for Regulation Area %s: %.4f\n', regulation_area, mse);
fprintf('Mean Absolute Error for Regulation Area %s: %.4f\n', regulation_area, mae);
fprintf('R-squared for Regulation Area %s: %.4f\n', regulation_area, r_squared);

% Visualize the results
figure;
scatter(y_test, y_pred, 'filled');
hold on;
plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--'); % Line y=x
title(['Regulation Area: ' regulation_area]);
xlabel('Actual Landings');
ylabel('Predicted Landings');
axis equal;
grid on;
hold off;

% User input for prediction
fprintf('Enter the following values for prediction:\n');
avg_temp = input('Average Temperature: ');
avg_oxy = input('Average Oxygen: ');
avg_chloro = input('Average Chlorophyll: ');
avg_salin = input('Average Salinity: ');
avg_year = input('Year: ');

% Create a new input array for prediction
user_input = [avg_temp, avg_oxy, avg_chloro, avg_salin, avg_year];

% Ensure mu and sigma are row vectors
mu = mu(:)'; % Transpose to ensure it's a row vector
sigma = sigma(:)'; % Transpose to ensure it's a row vector

% Standardize the user input
user_input_scaled = (user_input - mu) ./ sigma;

% Make prediction based on user input
predicted_landings = predict(knnModel, user_input_scaled);

% Display the prediction
fprintf('Predicted Landings for Regulation Area %s: %.4f\n', regulation_area, predicted_landings);
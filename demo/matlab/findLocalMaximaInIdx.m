function localMaxima = findLocalMaximaInIdx(matrix, idx, Target_number)
    % Get the size of the matrix
    [m, n] = size(matrix);

    % Initialize an empty array to store the coordinates and values of the local maxima
    localMaxima = [];

    % Iterate through each element in the matrix (ignoring boundary elements)
    for i = ceil(idx/2):m-floor(idx/2)
        for j = ceil(idx/2):n-floor(idx/2)
            % Extract the current idx x idx submatrix
            subMatrix = matrix(i-floor(idx/2):i+floor(idx/2), j-floor(idx/2):j+floor(idx/2));
            % Value of the current element
            currentValue = matrix(i, j);
            % Get the values of all neighbors in the idx x idx submatrix (excluding the center point)
            neighbors = subMatrix(:);
            centerIndex = sub2ind(size(subMatrix), ceil(idx/2), ceil(idx/2));
            neighbors(centerIndex) = []; % Exclude the center point

            % Check if the current element is a local maximum
            if all(currentValue > neighbors)
                localMaxima = [localMaxima; i, j, currentValue];
            end
        end
    end

    % If the number of found local maxima exceeds Target_number, keep only the top Target_number maxima
    if size(localMaxima, 1) > Target_number
        % Sort by value in descending order
        localMaxima = sortrows(localMaxima, -3);
        % Keep only the top Target_number local maxima
        localMaxima = localMaxima(1:Target_number, :);
    end
end
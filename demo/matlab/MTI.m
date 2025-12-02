function Cancel_PulseCompre_Data = MTI(framesData,cancelFlag,rangeFFTNum,pluseCancelSpace)
% Description: Perform clutter cancellation and pulse compression on a frame of radar echo data.

% Input:   
% framesData: Radar echo data requiring clutter cancellation and pulse compression
% cancelFlag: Clutter cancellation flag: cancelFlag=0 for mean cancellation; cancelFlag=1 for two-pulse cancellation; cancelFlag=2 for three-pulse cancellation
% rangeFFTNum: Number of FFT points for pulse compression
% pluseCancelSpace: Interval for two-pulse cancellation; only required if using two-pulse cancellation

% Output:
% Cancel_PulseCompre_Data: Complex data after pulse compression

%% Perform clutter cancellation and pulse compression on the data
meanCancel_mode = 0;
twopCancel_mode = 1;
thrpCancel_mode = 2;
numChirp = size(framesData,1);
%--------------------------------Mean Cancellation
if(cancelFlag == meanCancel_mode)
    meanCancelData = framesData - repmat(mean(framesData,1),[numChirp 1]); % Mean cancellation to remove DC component
    Cancel_PulseCompre_Data = fft(meanCancelData,rangeFFTNum,2); % Pulse compression X = ifft(Y,n,dim) returns the n-point inverse Fourier transform along dimension dim. For example, if Y is a matrix, then ifft(Y,n,2) returns the n-point inverse transform of each row.
end
%--------------------------------Two-Pulse Cancellation
if(cancelFlag == twopCancel_mode)
    twopCancelData = framesData(1:end-pluseCancelSpace,:,:) - framesData(1+pluseCancelSpace:end,:,:);
    twopCancelData(end+1,:,:) = framesData(end,:,:);
    Cancel_PulseCompre_Data = fft(twopCancelData,rangeFFTNum,2); % Pulse compression
end
%--------------------------------Three-Pulse Cancellation
if(cancelFlag == thrpCancel_mode)
    thrpCancelData = framesData(1:end-2,:,:) - 2 * framesData(2:end-1,:,:) + framesData(3:end,:,:);
    Cancel_PulseCompre_Data = ifft(thrpCancelData,rangeFFTNum,2); % Pulse compression
end
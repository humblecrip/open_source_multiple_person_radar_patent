%% Multi-Person Respiration and Heartbeat Detection Based on FMCW Radar
close all;
clc;
clear all;

%% Radar Parameters
% %% Radar Parameters
% % Parameter Settings
f0=60e8; % Start frequency
numADCSamples = 200; % Number of ADC samples
freqSlopeRate = 60.006e12; % Frequency slope rate, unit: MHz/us
adcSampleRate = 4e6; % Fast time ADC sampling rate, unit: Ksps
% numChirps = 128; % Number of chirps per frame
Ts = 50e-3; % Time between each frame
Tc = numADCSamples/adcSampleRate;
% Tc = 5.7e-5;
B = Tc*freqSlopeRate;%The actual Bandwidth.
deltaR = 3e8/(2*B);
d=0.025;
lambda=2*d;
%% Import Data
% load two_people_1.mat channel_data  % 1200x200x8 (taking one chirp per frame) frame sampling points

% Step 1: Read the data from the .mat file
loadedData = load('two_people_1.mat');

% Step 2: View the variable names in the loaded data
variableNames = fieldnames(loadedData);

% Step 3: Save the data as a new custom variable name
% Assume there is only one variable in the .mat file
channel_data = variableNames{1};

% New custom variable name
channel_data = loadedData.(channel_data);

% Check the data of the custom variable name
% disp(myCustomVarName);
useFramesNum=1200;
rangeFFTNum=256;
%% Data Processing
loop_cnt = floor(length(channel_data(:,1,1)) / useFramesNum);

% 初始化结果存储数组：[segment, target, HR_radar, RR_radar]
all_results = [];

for k = 1:loop_cnt
    
    use_channel_data = channel_data((k - 1) * useFramesNum + 1 : k * useFramesNum,:,:);
    
    %% Mean Cancellation and Pulse Compression
    % rangeProfile = MTI_PulseCompression(use_channel_data,0,rangeFFTNum); 
    % rangeProfile = MTI(use_channel_data,1,rangeFFTNum,1); % 1024x256x8 complex double 
    rangeProfile = MTI(use_channel_data,0,rangeFFTNum);
    sum_rangProfile = sum(abs(rangeProfile(:,:,1)),1);
    % sum_rangProfile([1:6 249:256])=0;
    [~,targetIndex] = max(sum_rangProfile);
    % targetIndex=40;
    % rangeProfile=fft(use_channel_data,256,2);
    %% DOA Estimation
    searchAngleRange = 60; % Angle spectrum search range is +-searchAngleRange degrees
    
    [~,azimuSpectrogram,Rxv] = IWR6843ISK_DOA(rangeProfile,2,useFramesNum,searchAngleRange);
    azimuSpectrogram=flipud(azimuSpectrogram);
    % azimuSpectrogram(:,100:end)=0;
    %% Finding Targets at Different Angles
    maxAzimu = max(azimuSpectrogram,[],2); % 121x1 max(A,[],2) returns a column vector containing the maximum value of each row

   % [values,peaks_index] = findpeaks(maxAzimu,'minpeakheight',3000000); % findpeaks(maxAzimu);
    [values,peaks_index] = findpeaks(maxAzimu,'minpeakheight',mean(azimuSpectrogram,'all'));
    maxAzimu=sum(azimuSpectrogram,2);
   figure(1);  
   plot(1:length(maxAzimu),10*log10(maxAzimu));    
   hold on; grid on
   plot(peaks_index,10*log10(maxAzimu(peaks_index)),'bd');
   xlabel('Angle/°');ylabel('Gain (dB)');title('MVDR Angle Estimation');

   hold off
   
   figure(2);

   % azimuSpectrogram(:,55:end)=0;
   data_max=findLocalMaximaInIdx(azimuSpectrogram,45,2);
   % data2=sum( azimuSpectrogram,1)
   % azimuSpectrogram(:,[1:6 249:256])=0;
   imagesc(-searchAngleRange:searchAngleRange,(1 :rangeFFTNum)*deltaR,abs(azimuSpectrogram.'));
   ylabel('Distance (m)');xlabel('Angle (°)');title('Range-Angle Spectrum');
   axis xy

   %% Range-Angle Fan Plot Drawing
    figure(3);
    R = (1 :rangeFFTNum)*deltaR;
    ang_ax =-searchAngleRange:searchAngleRange;
    X = R'*cosd(ang_ax); Y = R'*sind(ang_ax); %
    pcolor(Y,X,abs(azimuSpectrogram.'));
    axis equal tight  % x-axis unit scale and y-axis unit scale length are equal, best reflecting the actual curve;
    shading interp % Shading, making the colors transition smoothly
    axis off
    initialAz = -90; endAz = 90; % Label text
    text((max(R)+10)*cosd(initialAz),(max(R))*sind(initialAz),...
    [num2str(initialAz) '^o']);
    text((max(R)+10)*cosd(endAz),(max(R))*sind(endAz),...
    [num2str(endAz) '^o']);
    text((max(R)+10)*cosd(0),(max(R))*sind(0),[num2str(0) '^o']);

   %% Select the Top Two Values with the Largest Magnitude in Angle
   [values_sort,index] =sort(values,'descend');
   peaks_index_sort =peaks_index(index);
   peaks_index_max =peaks_index_sort([1,5]);

   %% Calculate Respiration and Heartbeat of Different Targets
   target_rangeProfile  = zeros(useFramesNum,length(peaks_index_max)); % Store the range data of the target after angle filtering, the subsequent respiration and heartbeat are processed based on this data, one column is the range data of one target  
    target_rangeProfile  = zeros(useFramesNum,size(data_max,1));
    for i=1:size(data_max,1)
        xt = squeeze(rangeProfile(:,data_max(i,2)-1,:));
        detAngle = -searchAngleRange + data_max(i,1) * (searchAngleRange * 2 / length(azimuSpectrogram(:,1)));

        fai = 2 * pi * sin(detAngle / 180 * pi) * d / lambda;

        aTheta = [1,exp(-1j*1*fai),exp(-1j*2*fai),exp(-1j*3*fai),exp(-1j*4*fai),exp(-1j*5*fai),exp(-1j*6*fai),exp(-1j*7*fai)].';%

        Wopt = (Rxv  * aTheta) / (aTheta' * Rxv  * aTheta);

        target_rangeProfile(:,i) = xt * Wopt; % target_rangeProfile 1024x2 loops twice
    end

   % xt = squeeze(rangeProfile(:,targetIndex,:)); % Extract 1024x8 complex double from 8 channels at each targetIndex=49, rangeProfile(:,targetIndex,2), rangeProfile(:,targetIndex,3), rangeProfile(:,targetIndex,6), rangeProfile(:,targetIndex,7)
   % 
   %  for m = 1:length(peaks_index_max)
   %      detAngle = -searchAngleRange + peaks_index_max(m) * (searchAngleRange * 2 / length(azimuSpectrogram(:,1)));
   % 
   %      fai = 2 * pi * sin(detAngle / 180 * pi) * d / lambda;
   % 
   %      aTheta = [1,exp(-1j*1*fai),exp(-1j*2*fai),exp(-1j*3*fai),exp(-1j*4*fai),exp(-1j*5*fai),exp(-1j*6*fai),exp(-1j*7*fai)].';%
   % 
   %      Wopt = (Rxv  * aTheta) / (aTheta' * Rxv  * aTheta);
   % 
   %      target_rangeProfile(:,m) = xt * Wopt; % target_rangeProfile 1024x2 loops twice
   %  end
   % 

    %% Estimate the Respiration and Heartbeat of Multiple Targets Sequentially, and Plot
    [breathRate,heartRate] = get_heartBreath_rate(target_rangeProfile,1/Ts);

    %% Output Results
    fprintf('\n========== Segment %d Results ==========\n', k);
    for i = 1:length(breathRate)
        fprintf('Target %d: Breath Rate = %d breaths/min, Heart Rate = %d beats/min\n', ...
            i, breathRate(i), heartRate(i));
        
        % 收集结果到数组：[segment, target, HR_radar, RR_radar]
        all_results = [all_results; k, i, heartRate(i), breathRate(i)];
    end
    fprintf('=========================================\n');

end

%% 导出 CSV 文件
fprintf('\n============================================\n');
fprintf('导出结果到 CSV 文件...\n');
fprintf('============================================\n');

% 将结果转换为表格
results_table = array2table(all_results, ...
    'VariableNames', {'segment', 'target', 'HR_radar', 'RR_radar'});

% 导出为 CSV 文件（保存在当前 MATLAB 工作目录）
csv_filename = 'radar_vital_signs_results.csv';
writetable(results_table, csv_filename);

fprintf('结果已保存到: %s\n', csv_filename);
fprintf('共 %d 条记录\n', size(all_results, 1));
fprintf('============================================\n\n');

% 显示汇总表格
disp('汇总结果:');
disp(results_table);

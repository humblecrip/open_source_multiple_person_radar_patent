clear all
close all
clc
T = readtable('D:\A_Data_in_brief\Dataset_source\1_AsymmetricalPosition\2_Log_data\Target1\position_ (1)\log_Target1_2_5GHZ_position1_ (1).csv');%Change to your own file path location.
T_frame=0.008;%The sampling interval of the Log data.
ECG=-1*T.Column1;
PCG=T.Column2;
ECG=ECG(end-3499:end);
PCG=PCG(end-3499:end);
threshold=20;%Set the threshold by the actual heatbeat waveform.
figure(1);
plot(T_frame*(1:length(ECG)),ECG(1:end),'b','linewidth',0.8);
xlabel('Time(s)','fontname','times new roman','fontsize',12);
ylabel('Ampulitude','fontname','times new roman','fontsize',12)
hold on
c = threshold;
lim=[0 T_frame*length(ECG)];
xlim(lim);
line(lim, [c c]); %Draw the heartbeat count baseline.
plot(T_frame*(1:length(ECG)),threshold,'linewidth',0.8);
lgd1=legend('Heatbeat Waveform','Heartbeat Count Baseline');
set(lgd1,'fontname','times new roman','fontsize',12);
hold off;

ys = smooth(PCG, 50, 'loess');
for i=1:800
    ys = smooth(ys, 50, 'loess');
end
[bre_peaks, bre_locs] = findpeaks(ys);%Get the actual number of heatbeats.
figure(2);
plot(T_frame*(1:length(PCG)),PCG);%Plot the respiratory waveform before the smooth filtering.
xlim([0 T_frame*length(PCG)]);
hold on;
plot(T_frame*(1:length(PCG)),ys,'LineWidth',2);%Plot the respiratory waveform after the smooth filtering.
for i=1:length(bre_locs)
    plot(T_frame*bre_locs(i),bre_peaks(i),'Color','b','Marker','v','MarkerFaceColor','b');
    hold on
end
xlabel('Time(s)','fontname','times new roman','fontsize',12);
ylabel('Ampulitude','fontname','times new roman','fontsize',12)
lgd2=legend('Respiratory waveforms without smoothing filtering','Respiratory waveforms with smoothing filtering','Respiration count');
set(lgd2,'fontname','times new roman','fontsize',12);

[hr_peaks, hr_locs] = findpeaks(ECG, 'MinPeakHeight', threshold);%Get the actual number of respirations.
fprintf('The respiratory count is %.d. \n',length(bre_peaks));%Output the respiratory count.
fprintf('The heatbeat count is %.d. \n',length(hr_peaks));%Output the heartbeat count.


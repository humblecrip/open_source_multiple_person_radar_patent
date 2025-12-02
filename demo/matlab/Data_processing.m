clear all
close all
clc
adcData=readDCA1000('adc_3GHZ_position1_ (1).bin',200,4);%Change the filename
numADCSamples=200;
numChirps=1200;
virsual_Rx=12;

data=zeros(numChirps,numADCSamples,virsual_Rx);
for i=1:virsual_Rx
    data(:,:,i)=reshape(adcData(i,:) , numADCSamples,numChirps ).';
    % data(:,:,i)=data(:,:,i).';
end
data(:,:,5:8)=[];%Obtain the Tx1 and Tx3's data only.
save two_people_1.mat data; %Save the data named two_people_1.mat.
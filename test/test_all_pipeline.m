%% test extraction features

clc; clear all; close all;

%% upload the rosneuro files
ros_buffer = load('/home/paolo/cvsa_ws/src/processing_cvsa/test/test_buffer.csv');
ros_filtered = load('/home/paolo/cvsa_ws/src/processing_cvsa/test/test_filtered_2.csv'); 
ros_features = load('/home/paolo/cvsa_ws/src/processing_cvsa/test/test_features.csv');
ros_features_received = load('/home/paolo/cvsa_ws/src/qda_cvsa/test/features_sended.csv');
ros_features_extracted = load('/home/paolo/cvsa_ws/src/qda_cvsa/test/features_extracted.csv');

%% matlab pipeline
file = '/home/paolo/prova39ch.gdf';
nchannels = 39;
frameSize = 32;
bufferSize = 512;
bands = {{10.0 12.0}, {14.0, 16.0}};
nbands = length(bands);
filterOrder = 4;
sampleRate = 512;
alignment = 1; % input('Input where there is the alignment from rosneuro and matlab: '); % 3sec + 1
selchs = {{'P5', 'PO5', 'PO7'}, {'P6', 'PO6', 'PO8'}};
% selchs = {{'P5'}, {'P6'}};
channels_label = {'FP1', 'FP2', 'F3', 'FZ', 'F4', 'FC1', 'FC2', 'C3', 'CZ', 'C4', 'CP1', 'CP2', 'P3', 'PZ', 'P4', 'POZ', 'O1', 'O2', 'EOG', ...
        'F1', 'F2', 'FC3', 'FCZ', 'FC4', 'C1', 'C2', 'CP3', 'CP4', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'PO7', 'PO8', 'OZ'};

[s,h] = sload(file);
s = s(:,1:nchannels);

nsamples = size(s,1);
nchunk = floor(nsamples/frameSize);
buffer = nan(bufferSize, nchannels);

zi_low = cell(1, nbands);
zi_high = cell(1, nbands);

buffer_all = [];
filtered_all = [];
dfet_total = [];
features_all = [];

for k = 1:nchunk
    frame = s((k-1)*frameSize+1:k*frameSize,:);
    buffer(1:end-frameSize,:) = buffer(frameSize+1:end,:);
    buffer(end-frameSize+1:end, :) = frame;

    % check
    if any(isnan(buffer))
        continue;
    end

    buffer_all = cat(1, buffer_all, buffer(:,1));

    dfet = [];
    features = [];

    for idx_band = 1:nbands
        c_zi_low = zi_low{idx_band};
        c_zi_high = zi_high{idx_band};

        c_band = cell2mat(bands{idx_band});
        [b_low, a_low] = butter(filterOrder, c_band(2) *(2/sampleRate), 'low');
        [b_high, a_high] = butter(filterOrder, c_band(1) *(2/sampleRate),'high');

        % take index of interest
        selch = selchs{idx_band};
        idx_interest_ch = zeros(1, numel(selch));
        for j=1:numel(selch)
            idx_interest_ch(j) = find(strcmp(channels_label, selch{j}));
        end

        % apply low and high pass filters
        [s_low, zi_low1] = filter(b_low,a_low,buffer,c_zi_low);
        [tmp_data,zi_high1] = filter(b_high,a_high,s_low,c_zi_high);

        if idx_band == 2
            filtered_all = cat(1, filtered_all, tmp_data(:,1));
        end

        zi_low{idx_band} = zi_low1;
        zi_high{idx_band} = zi_high1;

        % apply pow
        tmp_data = power(tmp_data, 2);

        % apply average
        tmp_data = mean(tmp_data, 1);

        % apply log
        tmp_data = log(tmp_data);

        dfet = cat(2, dfet, tmp_data(idx_interest_ch));
        features = cat(2, features, tmp_data);
    end

    dfet_total = cat(1, dfet_total, dfet);
    features_all = cat(1, features_all, features);
% 
    if size(filtered_all, 1) > (size(ros_filtered, 1) + 512*alignment)
        break;
    end
end

%% Plot di comparazione per features extracted
matlab_log = dfet_total;
channelId = 1;
alinment_features = alignment + 1;

figure;
subplot(2, 1, 1);
hold on;
plot(ros_features_extracted(:, channelId), 'b', 'LineWidth', 1);
plot(matlab_log(alinment_features:size(ros_features_extracted, 1)+alinment_features-1, channelId), 'r');
legend('rosneuro', 'matlab');
hold off;
grid on;

subplot(2,1,2)
bar(abs(ros_features_extracted(:, channelId)- matlab_log(alinment_features:size(ros_features_extracted, 1)+alinment_features-1, channelId)));
grid on;
xlabel('time [s]');
ylabel('amplitude [uV]');
title('Difference')

sgtitle('Evaluation features extracted' )

%% Plot di comparazione per features received
matlab_log = features_all;
channelId = 5;

figure;
subplot(2, 1, 1);
hold on;
plot(ros_features_received(:, channelId), 'b', 'LineWidth', 1);
plot(matlab_log(alinment_features:size(ros_features_received, 1)+alinment_features-1, channelId), 'r');
legend('rosneuro', 'matlab');
hold off;
grid on;

subplot(2,1,2)
bar(abs(ros_features_received(:, channelId)- matlab_log(alinment_features:size(ros_features_received, 1)+alinment_features-1, channelId)));
grid on;
xlabel('time [s]');
ylabel('amplitude [uV]');
title('Difference')

sgtitle('Evaluation features received' )

%% Plot di comparazione per features
matlab_log = features_all;
channelId_matlab = 29;
channelId_ros = 28;

figure;
subplot(2, 1, 1);
hold on;
plot(ros_features(:, channelId_ros), 'b', 'LineWidth', 1);
plot(matlab_log(alinment_features:size(ros_features, 1)+alinment_features-1, channelId_matlab), 'r');
legend('rosneuro', 'matlab');
hold off;
grid on;

subplot(2,1,2)
bar(abs(ros_features(:, channelId_ros)- matlab_log(alinment_features:size(ros_features, 1)+alinment_features-1, channelId_matlab)));
grid on;
xlabel('time [s]');
ylabel('amplitude [uV]');
title('Difference')

sgtitle('Evaluation features' )

%% Plot di comparazione per buffer

matlab_buffer = buffer_all;
alignment = 512*alignment+1; % input('Input where there is the alignment from rosneuro and matlab: '); % 3sec + 1
channelId = 1;

figure;
subplot(2, 1, 1);
hold on;
plot(ros_buffer(:, channelId), 'b', 'LineWidth', 1);
plot(matlab_buffer(alignment:size(ros_buffer, 1)+alignment-1, channelId), 'r');
legend('rosneuro', 'matlab');
hold off;
grid on;

subplot(2,1,2)
bar(abs(ros_buffer(:, channelId)- matlab_buffer(alignment:size(ros_buffer, 1)+alignment-1, channelId)));
grid on;
xlabel('time [s]');
ylabel('amplitude [uV]');
title('Difference')

sgtitle('Evaluation buffers' )


%% Plot di comparazione per filtered

matlab_filtered = filtered_all;
rosneuro = ros_filtered;

figure;
subplot(2, 1, 1);
hold on;
plot(rosneuro(:, channelId), 'b', 'LineWidth', 1);
plot(matlab_filtered(alignment:size(rosneuro, 1)+alignment-1, channelId), 'r');
legend('rosneuro', 'matlab');
hold off;
grid on;

subplot(2,1,2)
bar(abs(rosneuro(:, channelId)- matlab_filtered(alignment:size(rosneuro, 1)+alignment-1, channelId)));
grid on;
xlabel('time [s]');
ylabel('amplitude [uV]');
title('Difference')

sgtitle('Evaluation filtered' )


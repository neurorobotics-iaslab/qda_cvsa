%% test qda for matlab, work only for evaluations
clc; clear all; close all;

addpath('/home/paolo/bci_vr_ws/src/analysis_cvsa/equal_ros')

%% Load features and prediction of ros
datapath = './src/qda_bci/';
features = load([datapath 'test/processed_data.csv']);
ros_prob = load([datapath 'test/classified.csv']);

yaml_QDA_path = [datapath 'test/qda_test.yaml'];
qda = loadQDA(yaml_QDA_path);

bands_qda = qda.bands;
nbands_qda = size(bands_qda, 1);
features_band = [8 14]; % publish test publish only 8-14 as band

baseline = 0.0; % baseline by default is 0.0

prob_all = [];
for idx_feature = 1:length(features)
    for idx_band = 1:nbands_qda
        if all(bands_qda(idx_band,:) == features_band)
            c_features = log(features(idx_feature, qda.idchans{idx_band})) - baseline;
            c_prob = apply_qda(qda, c_features);
            prob_all = cat(1, prob_all, c_prob);
        end
    end
end


matlab_prob = prob_all(3:end,:); % align in ros we loose the first 3 messages

min_length = min(size(ros_prob,1), size(matlab_prob,1));

max_1 = max(abs(ros_prob(:,1) - matlab_prob(1:min_length,1)));
max_2 = max(abs(ros_prob(:,2) - matlab_prob(1:min_length,2)));

figure();
subplot(2, 2, 1);
plot(1:size(ros_prob,1), ros_prob(:,1), 'Color', 'r');
hold on;
plot(1:size(ros_prob,1), matlab_prob(1:min_length,1), 'Color', 'b');
title('First class for both methods')
legend('ros probs', 'matlab probs')

subplot(2, 2, 3);
plot(1:size(ros_prob,1), ros_prob(:,1)- matlab_prob(1:min_length,1), 'Color', 'r');
hold on;
text(0.05, 0.95, ['max diff: ' num2str(max_1)], 'Units', 'normalized');
title('Diff ros and matlab')
legend('Diff')


subplot(2, 2, 2);
plot(1:size(ros_prob,1), ros_prob(:,2), 'Color', 'r');
hold on;
plot(1:size(ros_prob,1), matlab_prob(1:min_length,2), 'Color', 'b');
title('Second class for both methods')
legend('ros probs', 'matlab probs')

subplot(2, 2, 4);
plot(1:size(ros_prob,1), ros_prob(:,1) - matlab_prob(1:min_length,1), 'Color', 'r');
hold on;
text(0.05, 0.95, ['max diff: ' num2str(max_2)], 'Units', 'normalized');
title('Diff ros and matlab')
legend('Diff')

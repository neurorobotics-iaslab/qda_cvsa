%% test qda for matlab, work only for evaluations
% clc; clear all; close all;


%% Load features and prediction of ros
features = load('/home/paolo/cvsa_ws/src/qda_cvsa/test/features_extracted.csv');
ros_prob = load('/home/paolo/cvsa_ws/src/qda_cvsa/test/ros_probs.csv');
r_all_features = load('/home/paolo/cvsa_ws/src/qda_cvsa/test/features_sended.csv');

yaml_QDA_path = '/home/paolo/cvsa_ws/src/qda_cvsa/test/qda_h7.yaml';
qda = loadQDA(yaml_QDA_path);

prob_all = [];
for idx_feature = 1:length(features)
    c_features = features(idx_feature,:);
    c_prob = apply_qda(qda, c_features);
    prob_all = cat(1, prob_all, c_prob);
end

max(ros_prob - prob_all, [], 'all')

figure();
subplot(2, 1, 1);
plot(1:size(ros_prob,1), ros_prob(:,1), 'Color', 'r');
hold on;
plot(1:size(ros_prob,1), prob_all(:,1), 'Color', 'b');
title('Difference for the first class')
legend('ros probs', 'matlab probs')

subplot(2, 1, 2);
plot(1:size(ros_prob,1), ros_prob(:,2), 'Color', 'r');
hold on;
plot(1:size(ros_prob,1), prob_all(:,2), 'Color', 'b');
title('Difference for the second class')
legend('ros probs', 'matlab probs')
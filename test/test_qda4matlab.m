%% test qda for matlab, work only for evaluations
clc; clear all; close all;

%% informations
c_subject = 'h7';
classes = [730 731];

bufferSize_feedback = 80;
frameSize = 32;
bufferSize_data = 512;
filterOrder = 4;
sampleRate = 512;

yaml_QDA_path = ['/home/paolo/cvsa_ws/src/qda_cvsa/cfg/qda_' c_subject '.yaml'];
qda = loadQDA(yaml_QDA_path);

bands = qda.bands;
selchs = qda.chans;
idx_selchs = qda.idchans;

channels_label = {'FP1', 'FP2', 'F3', 'FZ', 'F4', 'FC1', 'FC2', 'C3', 'CZ', 'C4', 'CP1', 'CP2', 'P3', 'PZ', 'P4', 'POZ', 'O1', 'O2', 'EOG', ...
        'F1', 'F2', 'FC3', 'FCZ', 'FC4', 'C1', 'C2', 'CP3', 'CP4', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'PO7', 'PO8', 'OZ'};

% load the files
path = ['/home/paolo/cvsa_ws/record/' c_subject '/gdf'];
files = dir(fullfile(path, '*evaluation*.gdf'));

features_all = [];
prob_all = [];
trials_start = [0];
final_prob = [];

%% take only interested data
for idx_f = 1:length(files)
    file = fullfile(path, files(idx_f).name);
    disp(['file (' num2str(idx_f) '/' num2str(length(files))  '): ', file])

    [signal,header] = sload(file);
    nchannels = length(channels_label);

    %% labeling
    disp('   Labelling')
    signal = signal(:,1:nchannels);
    events = header.EVENT;
    cuePOS = events.POS(events.TYP == 730 | events.TYP == 731);
    cueDUR = events.DUR(events.TYP == 730 | events.TYP == 731);
    cueTYP = events.TYP(events.TYP == 730 | events.TYP == 731);
    cfDUR  = events.DUR(events.TYP == 781);
%     cuePOS = cuePOS(3:end) - 1;
%     cueTYP = cueTYP(3:end);
%     cueDUR = cueDUR(3:end);
    cfPOS  = events.POS(events.TYP == 781)-1;
    hit_miss_timeout = events.TYP(events.TYP == 899 | events.TYP == 897 | events.TYP == 898);
    nTrials = length(cueTYP);

    %% Initialization variables
    disp('   Initialization variables')
    
    zi_high = cell(1, length(bands));
    zi_low  = cell(1, length(bands));
    for i = 1:length(bands)
        zi_high{i} = [];
        zi_low{i}  = [];
    end

    %% iterate over trials
    for i = 1:nTrials
        disp(['      trial ' num2str(i) '/' num2str(nTrials)])
        % initialization variables
        buffer = nan(bufferSize_data, nchannels);
        start_trial = cuePOS(i);
        end_trial = cfPOS(i) + cfDUR(i) - 1;
        % division for frameSize
        end_trial = int64(ceil(single(end_trial-start_trial)/32)*32) + start_trial;
        data = signal(start_trial:end_trial,:);
        % initialization variables for accumulation framework
        accumulator = repmat([qda.classes(1), qda.classes(2)], 1, bufferSize_feedback/qda.nclasses);
        idx_buffer_feedback = 1;

        nchunks = (end_trial-start_trial) / 32;
        for j = 1:nchunks
            frame = data((j-1)*frameSize+1:j*frameSize,:);
            buffer(1:end-frameSize,:) = buffer(frameSize+1:end,:);
            buffer(end-frameSize+1:end, :) = frame;

            % check
            if any(isnan(buffer))
                continue;
            end

            % iterate over bands
            features = nan(1, qda.nfeatures);
            for idx_band = 1:length(bands)
                c_band = bands(idx_band,:);
                [c_b_low, c_a_low] = butter(filterOrder, c_band(2)*(2/sampleRate),'low');
                [c_b_high, c_a_high] = butter(filterOrder, c_band(1)*(2/sampleRate),'high');
                c_zi_low = zi_low{idx_band};
                c_zi_high = zi_high{idx_band};

                % apply low and high pass filters
                [s_low, c_zi_low] = filter(c_b_low,c_a_low,buffer,c_zi_low);
                [tmp_data,c_zi_high] = filter(c_b_high,c_a_high,s_low,c_zi_high);
                zi_low{idx_band} = c_zi_low;
                zi_high{idx_band} = c_zi_high;

                % apply pow
                tmp_data = power(tmp_data, 2);

                % apply average
                tmp_data = mean(tmp_data, 1);

                % apply log
                tmp_data = log(tmp_data);

                idx_features = idx_band; % we repeat the bands to have same index for band and features
                features(idx_features) = tmp_data(idx_selchs(idx_features));
            end

            % apply qda
            c_prob = apply_qda(qda, features);
            features_all = cat(1, features_all, features);
            
            % apply the integrator
            if c_prob(1) > c_prob(2)
                accumulator(idx_buffer_feedback) = qda.classes(1);
            else
                accumulator(idx_buffer_feedback) = qda.classes(2);
            end
            
            if idx_buffer_feedback == bufferSize_feedback
                idx_buffer_feedback = 1;
            else
                idx_buffer_feedback = idx_buffer_feedback + 1;
            end

            % check in the integrator buffer the value of each class
            prob = [sum(accumulator == qda.classes(1)), sum(accumulator == qda.classes(2))] ./ bufferSize_feedback;
            prob_all = cat(1, prob_all, c_prob);
        end
        trials_start =cat(1, trials_start, length(prob_all));
        disp(['         must be a class: ' num2str(cueTYP(i)) ' and an ' num2str(hit_miss_timeout(i))])
        disp(['            final probs: ' num2str(prob(1)) '-' num2str(prob(2)) '. (730-731)'])
        final_prob = cat(1, final_prob, prob);
    end

end

save('features_extracted_4qdatest.mat', 'features_all', 'trials_start');

%% check with python predictions
qda_python = load('qda_prediction.mat');
qda_python = qda_python.qda_prediction;
max(qda_python - prob_all, [], 'all')

qda_integrated = load('final_4trial.mat');
qda_integrated = qda_integrated.final_4trial;
max(qda_integrated - final_prob, [], 'all')
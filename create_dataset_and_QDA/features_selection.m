%% Run for the dataset creation. It shows the logband, and the fisher to select the features
close all; clear all; clc;

%% Define variables
% file info
c_subject = 'h8'; % input
prompt = 'Enter "calibration" or "evaluation": ';
test_typ = input(prompt, 's');
day = '/20241015'; % input

path = ['/home/paolo/cvsa_ws/record/' c_subject day];
path_gdf = [path '/gdf/' test_typ];
chanlocs_path = '/home/paolo/new_chanlocs64.mat';

if strcmp(test_typ, "evaluation")
    feature_file = [path '/dataset/fischer_scores_ev.mat'];
    logband_file = [path '/dataset/logband_power_ev.mat'];
else
    feature_file = [path '/dataset/fischer_scores.mat'];
    logband_file = [path '/dataset/logband_power.mat'];
end

classes = [730,731];
nclasses = length(classes);

load(chanlocs_path);
files = dir(fullfile(path_gdf, '*.gdf'));  

band = {[8 10], [10 12], [12 14], [14 16], [16 18], [8 14]};
nbands = length(band);

s=[]; events = struct('TYP',[],'POS',[],'SampleRate',512,'DUR',[]); Rk=[];

channels_label = {'', '', '', '', '', '', '', '', '', '', '', '', 'P3', 'PZ', 'P4', 'POZ', 'O1', 'O2', '', ...
       '', '', '', '', '', '', '', '', '', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'PO7', 'PO8', 'OZ'};
 
% channels_label = {'FP1', 'FP2', 'F3', 'FZ', 'F4', 'FC1', 'FC2', 'C3', 'CZ', 'C4', 'CP1', 'CP2', 'P3', 'PZ', 'P4', 'POZ', 'O1', 'O2', 'EOG', ...
%         'F1', 'F2', 'FC3', 'FCZ', 'FC4', 'C1', 'C2', 'CP3', 'CP4', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'PO7', 'PO8', 'OZ'};


%% Concatenate files
for i=1:length(files)
    file = fullfile(path_gdf, files(i).name);

    %load(file);
    [signal,header] = sload(file);
    
    curr_s = signal(:,1:39);
    curr_h = header.EVENT;
    
    if strcmp(test_typ, "calibration") || (strcmp(test_typ, "evaluation"))
        start = find(curr_h.TYP == 1,1,'first');
        curr_h.TYP = curr_h.TYP(start:end);
        curr_h.POS = curr_h.POS(start:end);
        curr_h.DUR = curr_h.DUR(start:end);
    end
    % Create Rk vector (run)
    cRk = i*ones(size(curr_s,1),1);
    Rk = cat(1,Rk,cRk);
    % Concatenate events
    events.TYP = cat(1, events.TYP, curr_h.TYP);
    events.DUR = cat(1, events.DUR, curr_h.DUR);
    events.POS = cat(1, events.POS, curr_h.POS + size(s, 1));
    s = cat(1, s, curr_s);
end

%% Processing
% Create Vector labels
[nsamples,nchannels] = size(s);
[feedb_pos, feedb_dur, fix_dur, fix_pos, cue_dur, cue_pos, ntrials] = extract_info_label(events, 781, 786, [730 731]);

% Extract trial data
[TrialStart, TrialStop, FixStart, FixStop, Ck, Tk] = extract_trial_info(s, events, fix_pos, fix_dur, feedb_pos, feedb_dur, cue_pos, ntrials);

% Applay the filtering
s_processed = NaN(nsamples,nchannels,nbands);
for f_idx=1:nbands
    sel_band = band{f_idx}; %Hz
    t_window = 1; %[s]
    windowSize = events.SampleRate*t_window;
    filtOrder = 4;
    s_movavg = data_processing(s, nchannels, events.SampleRate, sel_band, filtOrder, t_window);
    s_processed(:,:,f_idx) = s_movavg;
end

% Trial extraction
trial_dur = min(TrialStop-TrialStart);
trialData =  []; new_Rk = []; new_Ck = [];
dataforTrial = NaN(trial_dur,nchannels,nbands,ntrials);
tCk = zeros(ntrials,1);
for trId=1:ntrials
    cstart = TrialStart(trId);
    cstop = cstart + trial_dur - 1;

    dataforTrial(:,:,:,trId) = s_processed(cstart:cstop,:,:);

    c_Rk = Rk(cstart:cstop,1);
    new_Rk = cat(1,new_Rk,c_Rk);
    c_Ck = Ck(cstart:cstop);
    new_Ck = cat(1,new_Ck,c_Ck);
    tCk(trId) = unique(nonzeros(Ck(cstart:cstop)));
    trialData = cat(1, trialData, s_processed(cstart:cstop,:,:));
end

% % Baseline extraction
% minFix_dur = min(FixStop - FixStart);
% reference = NaN(minFix_dur, nchannels, nbands, ntrials);
% for trId=1:ntrials
%     cstart = FixStart(trId); %=TrialStart(trId) o fix_pos(trId)
%     cstop = cstart+ minFix_dur - 1;
%     reference(:,:,:,trId) = s_processed(cstart:cstop,:,:);
% end
% baseline = repmat(mean(reference),[size(dataforTrial,1) 1 1]);

%% Compute LogBandPOwer [samples x channels x bands] 
% ERD = log(dataforTrial./baseline); % use the 4th dim for trials
ERD = log(trialData); % all trials concatenated
ERD = permute(ERD, [1 3 2]); 

%% fisher score
% % compute for each run
% Runs = unique(Rk);
% nruns = length(Runs);
% fischer_score = NaN(nbands,nchannels,nruns);
% F2S = NaN(nbands*nchannels,nruns);
% for rId=1:nruns
%     rindex = new_Rk==Runs(rId);
%     cmu = NaN(nbands,nchannels,2);
%     csigma = NaN(nbands,nchannels,2);
% 
%     for cId=1:nclasses
%        cindex = rindex & new_Ck==classes(cId);
%        cmu(:,:,cId) = squeeze(mean(ERD(cindex,:,:)));
%        csigma(:,:,cId) = squeeze(std(ERD(cindex,:,:)));
%     end
%     fischer_score(:,:,rId) = abs(cmu(:,:,2)-cmu(:,:,1))./sqrt((csigma(:,:,1).^2 + csigma(:,:,2).^2));
% end
% % compute in general
% fisher_score_total = NaN(nbands,nchannels);
% cmu_total = NaN(nbands,nchannels,2);
% csigma_total = NaN(nbands,nchannels,2);
% for cId=1:nclasses
%        cindex_new = new_Ck==classes(cId);
%        cmu_total(:,:,cId) = squeeze(mean(ERD(cindex_new,:,:)));
%        csigma_total(:,:,cId) = squeeze(std(ERD(cindex_new,:,:)));
% end
% fischer_score_total(:,:) = abs(cmu_total(:,:,2)-cmu_total(:,:,1))./sqrt((csigma_total(:,:,1).^2 + csigma_total(:,:,2).^2));
% 
% % Visualization for each run
% disp('[proc] |- Visualizing cva for offline runs');
% freq_intervals = {'8-10', '10-12', '12-14', '14-16', '16-18'};
% OfflineRuns = unique(new_Rk);
% NumCols = length(OfflineRuns);
% climits = [];
% handles = nan(length(OfflineRuns), 1);
% a = find(~strcmp(channels_label,''));
% figure;
% colormap('jet');
% for rId = 1:length(OfflineRuns)
%     subplot(2, ceil(NumCols/2), rId);
%     imagesc(fischer_score(:, a, OfflineRuns(rId))');
%     axis square;
%     colorbar;
%     set(gca, 'XTick', 1:nbands);
%     set(gca, 'XTickLabel', freq_intervals);
%     set(gca, 'YTick', 1:size(a,2));
%     set(gca, 'YTickLabel', channels_label(find(~strcmp(channels_label,''))));
%     xtickangle(90);
%     xlabel('Hz');
%     ylabel('channel');
% 
%     title([ test_typ ' run ' num2str(OfflineRuns(rId))]);
% 
%     climits = cat(2, climits, get(gca, 'CLim'));
%     handles(OfflineRuns(rId)) = gca;
% end
% set(handles, 'clim', [0 max(max(climits))]);
% sgtitle(['Fisher score Subj: ' c_subject]);
% 
% % visualization in general
% figure;
% colormap('jet');
% imagesc(fischer_score_total(:,a)');
% axis square;
% colorbar;
% set(gca, 'XTick', 1:nbands);
% set(gca, 'XTickLabel', freq_intervals);
% set(gca, 'YTick', 1:size(a,2));
% set(gca, 'YTickLabel', channels_label(find(~strcmp(channels_label,''))));
% xtickangle(90);
% xlabel('Hz');
% ylabel('channel');
% title(['Total FS Subj: ' c_subject]);

%% CVA
% Compute cva for each run
Runs = unique(Rk);
nruns = length(Runs);
cva = nan(nbands, nchannels, nruns);
for idx_r = 1:nruns
    for i= 1:nbands
        rindex = new_Rk==Runs(idx_r);
        c_data = squeeze(ERD(rindex,i,:));
        c_ck = new_Ck(rindex);
        c = cva_tun_opt(c_data, c_ck);
        cva(i, :, idx_r) = c;
    end
end

% Compute in general
cva_total = NaN(nbands,nchannels);
for i= 1:nbands
    c_data = squeeze(ERD(:,i,:));
    c_ck = new_Ck;
    c = cva_tun_opt(c_data, c_ck);
    cva_total(i, :) = c;
end

% Visualization for each run
disp('[proc] |- Visualizing cva for offline runs');
freq_intervals = cell(size(band));
for i = 1:length(band)
    freq_intervals{i} = sprintf('%d-%d', band{i}(1), band{i}(2));
end
OfflineRuns = unique(new_Rk);
NumCols = length(OfflineRuns);
climits = [];
handles = nan(length(OfflineRuns), 1);
a = find(~strcmp(channels_label,''));
figure;
colormap('jet');
for rId = 1:length(OfflineRuns)
    subplot(2, ceil(NumCols/2), rId);
    imagesc(cva(:, a, OfflineRuns(rId))');
    axis square;
    colorbar;
    set(gca, 'XTick', 1:nbands);
    set(gca, 'XTickLabel', freq_intervals);
    set(gca, 'YTick', 1:size(a,2));
    set(gca, 'YTickLabel', channels_label(find(~strcmp(channels_label,''))));
    xtickangle(90);
    xlabel('Hz');
    ylabel('channel');

    title([ test_typ ' run ' num2str(OfflineRuns(rId))]);

    climits = cat(2, climits, get(gca, 'CLim'));
    handles(OfflineRuns(rId)) = gca;
end
set(handles, 'clim', [0 max(max(climits))]);
sgtitle(['CVA Subj: ' c_subject]);

% Visualization cva in general
figure;
colormap('jet');
imagesc(cva_total(:,a)');
axis square;
colorbar;
set(gca, 'XTick', 1:nbands);
set(gca, 'XTickLabel', freq_intervals);
set(gca, 'YTick', 1:size(a,2));
set(gca, 'YTickLabel', channels_label(find(~strcmp(channels_label,''))));
xtickangle(90);
xlabel('Hz');
ylabel('channel');
title(['Total CVA Subj: ' c_subject]);

%% Topoplot LogBand power
% as for all the topoplot it depends on the time required!
ERDfortrial = log(dataforTrial);
chanlocs_label = {chanlocs.labels};
% start_cue = input();
period = [3 (trial_dur/events.SampleRate)]*events.SampleRate; % start_cue - end_cf

%Select channels
sel_channels =  {'', '', '', '', '', '', '', '', '', '', '', '', 'P3', 'PZ', 'P4', 'POZ', 'O1', 'O2', '', ...
    '', '', '', '', '', '', '', '', '', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'PO7', 'PO8', 'OZ'};
cuetocf_erd1 = mean(mean(ERDfortrial(period(1):period(2), :, :, tCk == classes(1)), 4), 1);
cuetocf_erd2 = mean(mean(ERDfortrial(period(1):period(2), :, :, tCk == classes(2)), 4), 1);
cuetocf_erd = squeeze(cuetocf_erd2-cuetocf_erd1);
cuetoc_feed = zeros(64, nbands);
for i=1:length(chanlocs_label)
    for j = 1:length(sel_channels)
        if strcmpi(chanlocs_label{i}, sel_channels{j})
            if ~isnan(cuetocf_erd(j))
                cuetoc_feed(i,:) = cuetocf_erd(j,:);
            else
                cuetoc_feed(i,:) = 0;
            end
        end
    end
end

% % saving subject id
% currentPath = fileparts(mfilename('fullpath'));
% subj = [currentPath '/c_subject.mat'];
% save(subj,'c_subject', 'path');

% saving fischer score for UI
rowLabels = channels_label(find(~strcmp(channels_label,'')));
colLabels = freq_intervals;
cva_selected = cva_total(:,a)';
save(feature_file, 'cva_selected', 'rowLabels', 'colLabels','band');

% saving logband for UI
logbandPower = cuetoc_feed;
electrodePos = chanlocs;
save(logband_file,'logbandPower','electrodePos', 'band');

%% Launching UI for feature selection and creation of the dataset
UI_CVSA(path)
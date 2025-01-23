function s_movavg = data_processing(signal, nchannels, sampleRate, band, filtOrder, avg)

% Filter Parameters
[b_low, a_low] = butter(filtOrder, band(2)/(sampleRate/2), 'low');
[b_high, a_high] = butter(filtOrder, band(1)/(sampleRate/2), 'high');
% Apply filters
sfilt = zeros(size(signal));
for chId=1:nchannels
    %Filtfilt
    sfilt(:,chId) = filter(b_low,a_low,signal(:,chId));
    sfilt(:,chId) = filter(b_high,a_high,sfilt(:,chId));
end

%Squaring
s_rect = power(sfilt,2);
%Moving average
s_movavg = zeros(size(signal));
for chId=1:size(signal,2)
    s_movavg(:,chId) = filter(ones(1,avg*sampleRate)/avg/sampleRate, 1, s_rect(:,chId));
end

end

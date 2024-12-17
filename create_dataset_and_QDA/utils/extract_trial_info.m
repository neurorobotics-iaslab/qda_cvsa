function [TrialStart, TrialStop, FixStart, FixStop, Ck, Tk] = extract_trial_info(signal, header, fix_pos, fix_dur, feedb_pos, feedb_dur, cue_pos, trials)

    Ck = zeros(size(signal,1),1);
    Tk = zeros(size(signal,1),1);
    TrialStart = NaN(trials,1);
    TrialStop = NaN(trials,1);
    FixStart = NaN(trials,1);
    FixStop = NaN(trials,1);
    for trId=1:trials
        cstart = fix_pos(trId);
        cstop = feedb_pos(trId) + feedb_dur(trId) - 1;
        Ck(cstart:cstop) = header.TYP(cue_pos(trId)==header.POS);
        Tk(cstart:cstop) = trId;

        TrialStart(trId) = cstart;
        TrialStop(trId) = cstop;
        FixStart(trId) = cstart;
        FixStop(trId) = cstart+fix_dur(trId)-1;
    end
end

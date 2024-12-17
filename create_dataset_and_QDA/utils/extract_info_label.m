function [feedb_pos, feedb_dur, fix_dur, fix_pos, cue_dur, cue_pos, ntrials] = extract_info_label(header, feedb_event, fix_event, cue_events)

    feedb_dur = header.DUR(header.TYP == feedb_event);
    feedb_pos = header.POS(header.TYP == feedb_event);
    fix_pos = header.POS(header.TYP == fix_event);
    fix_dur = header.DUR(header.TYP == fix_event);
    cue_pos = header.POS(ismember(header.TYP, cue_events));
    cue_dur = header.DUR(ismember(header.TYP, cue_events));

    cue_pos = cue_pos(length(cue_pos)-length(feedb_pos)+1:end); % do this beacuse some value are in the eye calibration (we know they are at the beginning)
    cue_dur = cue_dur(length(cue_dur)-length(feedb_dur)+1:end);

    ntrials = sum(header.TYP == 781);   %length(feedb_pos);
end

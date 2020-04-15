

import numpy as np
import music21 as m21
from music21.note import Note
import math

import fractions

def ratio_to_halfsteps(rat):
    return 12 * math.log(rat, 2)

def gen_comma_meas(start_pitch,note_len):

    p5 = m21.interval.Interval(ratio_to_halfsteps(3/2))
    p8 = m21.interval.Interval(ratio_to_halfsteps(2))
    M3 = m21.interval.Interval(ratio_to_halfsteps(5/4))

    bass_notes = []
    bass_1_low_p =start_pitch
    bass_1_high_p = p5.transposePitch(bass_1_low_p)
    bass_notes.append( m21.chord.Chord(
                        [bass_1_low_p,
                         bass_1_high_p],quarterLength=note_len*2))

    treb_notes = []
    treb_1_p = p8.transposePitch(bass_1_low_p)
    treb_2_p = p5.transposePitch(bass_1_high_p)
    treb_notes.extend([
                    Note(treb_1_p,quarterLength=note_len),
                    Note(treb_2_p,quarterLength=note_len*2)
    ])


    bass_2_high_p = p5.transposePitch(p8.reverse().transposePitch(treb_2_p))
    bass_2_low_p = M3.reverse().transposePitch(bass_2_high_p)
    bass_notes.append(m21.chord.Chord(
                        [bass_2_low_p,
                         bass_2_high_p],quarterLength=note_len*2)
    )

    treb_3_p = p5.transposePitch(bass_2_low_p)
    treb_notes.append(Note(treb_3_p,quarterLength=note_len))

    end_pitch = p8.reverse().transposePitch(treb_3_p)

    bass_meas = m21.stream.Measure()
    treb_meas = m21.stream.Measure()

    for note in bass_notes:
        bass_meas.append(note)
    for note in treb_notes:
        treb_meas.append(note)

    # Following code attempted to make a shepard tone out of the comma pump.
    # Sound rendering was not good with current synths.
    # p15 = m21.interval.Interval('p15')
    # p31 = m21.interval.add([p15,p15])
    # extra_meas = m21.stream.Measure([bass_meas.transpose(p15),bass_meas.transpose(p15.reverse()),
                                     #  treb_meas.transpose(p15),treb_meas.transpose(p15.reverse()),
                                     # bass_meas.transpose(p31), bass_meas.transpose(p31.reverse()),
                                     # treb_meas.transpose(p31), treb_meas.transpose(p31.reverse()),
                                     # ])


    return (bass_meas,treb_meas),end_pitch

def make_comma_pump(start_pitch,note_len,n_measures):
    bass_part = m21.stream.Part()  # m21.instrument.Piano())
    treb_part = m21.stream.Part()  # m21.instrument.Piano())
    #ottava_part = m21.stream.Part()  # m21.instrument.Piano())

    for i in range(n_measures):
        (bass_meas, treb_meas), start_pitch = gen_comma_meas(start_pitch, note_len)
        bass_part.append(bass_meas)
        treb_part.append(treb_meas)  # ottava_part.append((ottava_meas))

    final_score = m21.stream.Score([treb_part, bass_part])  # ,ottava_part])
    return final_score


if __name__=='__main__':
    start_pitch = m21.pitch.Pitch('A3')
    note_len=1
    n_measures = 2

    final_score = make_comma_pump(start_pitch,note_len,n_measures)
    final_score.write('midi','comma_pump.midi')
    final_score.write('musicxml','comma_pump_min.xml')

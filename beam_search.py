''' Beam Search Implementation for writing a melody over a given chord progression. '''

import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.models import load_model

PATH = r'C:\Users\Holl\Documents\Promotion\Projekte\ML\Code Snippet'

MODEL = PATH + r'\melody_model.h5'
CHORDS = PATH + r"\chords.csv"
INPUT_MELODY = PATH + r'\melody.npy'

class BeamSearch:
    ''' Conduct beam search '''

    def __init__(self, model=MODEL, chords=CHORDS, input_melody=INPUT_MELODY,
                 k=3, alpha=.7, prob_scaling=.1, note_weight=.5):
        '''
        Args:
            [model]         = string: keras model (.h5 file)
            [chords]        = string: chords file (.csv file)
            [input_melody]  = string: file with input melody (.npy file)
            [k]             = beam_width
                              k = 1 --> greedy search
                              k > 1 --> beam search
            [alpha]         = alpha for length normalization:
                              1 / length(candidate) ** alpha * prob(candidate)
                              alpha = 1 --> complete length normalization
                              alpha = 0 --> no length normalization
            [prob_scaling]  = own method, so handle carefully!
                              probabilities of model are exponentiated with this factor
                              prob_scaling = 1 --> no scaling at all
                              prob_scaling towards 0 --> favors smaller probabilities
                              I.e. less 'obvious' candidates can be favored to some degree
                              Should probably not be smaller than .01 (rounding errors)
            [note_weight]   = float: gives the option to weight importance of
                              note value vs. note duration.
                              1-note_weight = weight of duration.
                              note_weight = .5 --> both weighted equally
                              note_weight = .8 --> note values weighted more heavily
                              note_weight = 1 --> duration values not weighted at all

        Returns :
            Nothing.          The property 'melody' stores the whole melody.  '''

        self.model      = load_model(model)
        self.seq_length = self.model.input[4].shape[1]
        self.chords     = pd.read_csv(chords)
        self.melody     = np.load(input_melody)
        self.k          = k
        self.alpha      = alpha
        self.scaling    = prob_scaling
        self.weighting  = note_weight
        self.no_measures= max(self.chords['Measure'])

    def write_beam_search_melody(self):
        ''' Implement beam search algorithm. Parameters are given by class. '''

        # Loop over all measures
        for i in tqdm(range(self.no_measures)):

            #print('Writing melody: Measure', i+1, '\t ( of ', self.no_measures, ')')

            # Initiate first candidate as final candidate from last measure
            cand_list = [
                    Candidate(value         = self.melody[:2, -1:].flatten(),
                              prob          = 0,
                              melody        = self.melody[:, -self.seq_length:],
                              cand_history  = [0, []],
                              measure       = i,
                              seq_length    = self.seq_length)]

            # List for candidates who reached end of measure
            finished_cands = []

            # Beam search until all candidates are finished.
            while continue_searching(cand_list):

                # First, store number of acitve candidates
                no_active_cands = len(cand_list)

                # Get k best candidates for each candidate in list
                for cand in cand_list[:no_active_cands]:
                    cand.predict_k_best(self.k, self.model, cand_list,
                                        self.chords, self.scaling, self.weighting)

                # Choose k best ones of the newly created candidates
                cand_list = update_candidates(cand_list, self.k, no_active_cands)

                # Store finished candidates in separate list
                [finished_cands.append(cand) for cand in cand_list if cand.finished]

                # Delete finished candidates from active candidate list
                cand_list = [cand for cand in cand_list if not cand.finished]

            final_cand = choose_final_cand(self.alpha, finished_cands)

            # Append the best melody to the previous melody
            self.melody = np.hstack((self.melody, final_cand[0].history))

        # Finally, remove input from beginning so that the created melody remains
        self.melody = self.melody[:, self.seq_length:]

class Candidate:
    ''' Candidate class. Contains all information needed for beam search.
        The Class works by instiating new instances when calling
        'predict_k_best'; these are appended to the list of candidates. '''

    def __init__(self, value, melody, prob, cand_history, measure, seq_length):
        ''' Args:
                [value]         = numpy array: (note_value, duration_value)
                [prob]          = numeric value; summed probability of previous
                                  log probabilities
                [melody]        = numpy array of length = seq_length.
                                  First row = note values
                                  second row = duration values
                                  third row = offset values
                                  --> needed as input to the RNN
                [cand_history]  = complete melody of candidate. Needed for when
                                  length(candidate) > seq_length
                [measure]       = integer; measure number. Needed for extracting the
                                  current chord
                [seq_length]    = the length of the sequence that the RNN takes as input.
                                  seq_length = 8 --> previous 8 notes as input to RNN. '''

        self.value      = value
        self.prob       = prob
        self.melody     = melody
        self.offset     = melody[-1, -1]
        self.next_offset = ((melody[1, -1:] + melody[2, -1:]) % 48)[0]
        self.measure    = measure
        self.seq_length = seq_length
        self.history    = cand_history[1]
        self.length     = cand_history[0]
        self.finished   = True if self.value[1] + self.offset >= 48 else False

    def get_curr_chord(self, chords):
        ''' Get the chord that the current note will be played over
        Args:
            [chords]    = chords file (pd DataFrame)
        Returns:
            [chord]     = np array of shape (4,) containing the four notes of the chord '''

        ch_meas = chords.loc[(chords['Measure'] == self.measure) &\
                             (chords['Offset'] <= (self.next_offset / 12))]
        min_diff = np.argmin(abs((self.next_offset / 12) - \
                                 np.array(ch_meas['Offset'])))
        chord   = ch_meas[ch_meas['No_in_meas'] == min_diff][['c1', 'c2', 'c3', 'c4']]
        chord   = np.array(chord).flatten() - 47
        return chord

    def predict_k_best(self, k, model, cand_list, chords, prob_scaling, weighting):
        ''' Predict k best next candidates for a candidate by running RNN.
        Args:
            [k]         = see above
            [model]     = see above
            [cand_list] = list to append k best candidates to
            [chords]    = see above
            [prob_scaling] = scaling of resulting probalities; see above
            [weighting] = weighting of value vs. duration; see above

        Returns:
            Nothing.      Functions appends k best candidates to the specified list '''

        # Transform the inputs to the objects that the neural network expects
        c_1, c_2, c_3, c_4  = self.get_curr_chord(chords)
        melody_input, dur_input, off_input = self.melody

        # RNN prediction
        model_pred = model.predict([c_1.reshape(1, 1),
                                    c_2.reshape(1, 1),
                                    c_3.reshape(1, 1),
                                    c_4.reshape(1, 1),
                                    melody_input.reshape(1, self.seq_length),
                                    dur_input.reshape(1, self.seq_length),
                                    off_input.reshape(1, self.seq_length) ])

        # Get k best candidates (both the values and their respective probabilities)
        # First, extract values from model output
        values = np.array([np.argsort(p.flatten())[-k:] for p in model_pred])

        # Clip values if they exceed the measure border
        values[1] = clip_end_of_measure(self.next_offset, values[1])

        # Extract probabilities
        probs = np.array([model_pred[i][:, values[i]].flatten() for i in range(2)])

        # Scale probabilities and take log probability; weight and sum them
        if prob_scaling != 1:
            probs = probs ** prob_scaling
            probs = weighting * probs[0] + (1-weighting) * probs[1] * self.prob
        else:
            probs = np.log(probs)
            probs = weighting * probs[0] + (1-weighting) * probs[1] + self.prob

        # For each of the k best, create a candidate object and append it to cand list
        for i in range(k):
            new_melody  = np.append(values[:, i], self.next_offset).reshape(3, 1)
            cand_hist   = np.hstack((self.history, new_melody)) if self.length > 0 else new_melody
            cand_list.append(
                    Candidate(value         = values[:, i],
                              prob          = probs[i],
                              melody        = np.hstack((self.melody,
                                                         new_melody))[:, -self.seq_length:],
                              cand_history  = [self.length+1, cand_hist],
                              measure       = self.measure,
                              seq_length    = self.seq_length)
                    )

def continue_searching(cand_list):
    ''' Return False if all candidates in candidate list are finished '''
    return False if len(cand_list) == 0 else True

def update_candidates(cand_list, k, no_active_cands):
    ''' Choose k best of a list of candidates while taking into account
        how many active candidates are still being developed

        Args:
            [cand_list] = list of active candidates
            [k] = see above
            [no_active_cands] = number of active candidates

        Returns:
            List with k best active candidates '''

    cand_list = cand_list[no_active_cands:]
    # Return k - no_finished except for the first iteration; here, return k
    if np.all([c.length == 1 for c in cand_list]):
        no_active_cands = k
    return sorted(cand_list, key=lambda x: x.prob, reverse=True)[:no_active_cands]

def choose_final_cand(alpha, finished_cands):
    ''' Implement length normalization

        Args:
            [alpha] = alpha (see formula above)
            [finished_cands] = list of finished candidates to choose from

        Returns:
            List with only the final candidate '''

    regularization = lambda x: 1 / x.length**alpha * x.prob
    return sorted(finished_cands, key = regularization, reverse=True)[:1]

def clip_end_of_measure(offset, durations):
    ''' Clips note values if they exceed the end of a measure

        Args:
            [offset] = integer; offset of note
            [durations] = np array of dim (k,)

        Returns:
            [durations] = np array of dim (k,) with clipped notes '''

    for i, dur in enumerate(durations):
        durations[i] = dur - (offset + dur - 48) if offset + dur > 48 else dur
    return durations

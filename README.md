# beam_search
Implementation of the Beam Search Algorithm for my Jazz Recurrent Neural Network. The algorithm takes a given chord structure (which was written by another Recurrent Neural Network) and writes a jazz melody on top of it using the Beam Search algorithm. 

The original algorithm has the following hyperparameters: 

* ```k```: beam width, i.e. number of nodes to explore in each step. 
* ```alpha```: used for length normalization. The formula for length normalization is ``` 1 / len(candidate) ** alpha * probability(candidate) ``` where the probability of the candidate is calculated as the sum of the log probabilities of all time steps T. If alpha=1, the probabilities are normalized completely. This prevents shorter sequences from being preferred simply due to their shorter length.

In order to have an even more flexible algorithm, I implemented a further hyperparameter: 
* ```prob_scaling```: the output probabilities of the Neural Network are exponentiated with this factor. If prob_scaling=1, the traditional algorithm is implemented, i.e. there is no scaling and the sum of the log probabilities are used. If prob_scaling<1, less 'obvious' candidates have more of a chance against the more obvious ones. In this case, not the sum of the log probabilities but the product of the regular probabilities is used as a measure of the probability of a candidate (because the problem of multiplying several very small numbers with each other is not as severe here).

Since the melody model has two outputs (note value (i.e. pitch frequency) and note duration), one can also choose to weight these with 
* ```note_weight```: If .5, both values and durations of the note are weighted equally; if .8, note values are weighted more heavily; if 1, only note values are taken into account.

The module can be used as follows:
```ruby
BeamSearch = BeamSearch(k=3, alpha=1, prob_scaling=1, note_weight=.5)
BeamSearch.write_beam_search_melody()
```
The final melody is then stored in 
```ruby
BeamSearch.melody
```
where ```BeamSearch.melody[0]``` contains the note values, ```BeamSearch.melody[1]``` contains the note durations and ```BeamSearch.melody[2]``` contains the note offsets. 

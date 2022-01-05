# self-shrinking-slope
Self shrinking of the sigmoid slope (4S), a.k.a. Prancing Pony, is a callback-based method that automatically finds a good slope hyper-parameter for your sigmoid/softmax.

Prancing Pony helps you skipping the boring hyper-parameter tuning of the slope of a sigmoid/softmax by embedding inside the trainig an automatic slope search. Not only Prancing Pony can lighten your work but it can also outperform tradional training strategies where the output critically depends on the slope.

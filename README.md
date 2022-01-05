# self-shrinking-slope
Self shrinking of the sigmoid slope (4S), a.k.a. Prancing Pony, is a callback-based method that automatically finds a good slope hyper-parameter for your sigmoid/softmax.

Prancing Pony helps you skipping the boring hyper-parameter tuning of the slope of a sigmoid/softmax by embedding inside the training an automatic slope search. Not only Prancing Pony can lighten your work but it can also outperform tradional training strategies where the output critically depends on the slope.

The callback is ready to use: you only have to dowload "callback.py"

A conference paper about Prancing Pony is currently under review and may soon be available.
We are currently working on a demo of Prancing Pony to be uploaded here on github.

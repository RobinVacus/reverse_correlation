# reverse_correlation
Contains the figures appearing in "Enhanced Food Availability can Deteriorate Fitness through Excessive Scrounging", as well as the code that was used to generate them.

To generate the figures yourself, run
```
python3 figures.py <resolution>
```
where 'resolution' indicates the number of values of each parameter for which the computations shall be performed.
If it is omitted, a small default resolution of 20 will be used.

Figures already present in the working directory will not be generated again -- they must be deleted manually, should they be created again with a different resolution.

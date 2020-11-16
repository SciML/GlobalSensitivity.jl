# eFAST Method

`eFAST` has `num_harmonics` as the only argument, it is the number of harmonics to sum in
the Fourier series decomposition and defaults to 4.

## eFAST Method Details

eFAST offers a robust, especially at low sample size, and computationally efficient procedure to
get the first and total order indices as discussed in Sobol. It utilizes monodimensional Fourier decomposition
along a curve exploring the parameter space. The curve is defined by a set of parametric equations,
```math
x_{i}(s) = G_{i}(sin ω_{i}s), ∀ i=1,2 ,..., n,
```
where s is a scalar variable varying over the range ``-∞ < s < +∞``, ``G_{i}`` are transformation functions
and ``{ω_{i}}, ∀ i=1,2,...,n`` is a set of different (angular) frequencies, to be properly selected, associated with each factor.
For more details on the transformation used and other implementation details you can go through [ A. Saltelli et al.](http://dx.doi.org/10.1080/00401706.1999.10485594).

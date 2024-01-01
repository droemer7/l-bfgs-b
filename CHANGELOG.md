# L-BFGS-B Change Log

## v2.0.0
* Removed `Function` template parameter from both `Solver` and `Lbfgsb`, passing a reference to the concrete `Function` throughout `Solver`. Note: C++17 is required if you don't want to have to write `Lbfgsb<>`.
* Renamed and repurposed `SolverState::reset` to `SolverState::stalled`, which means the solver produced the exact same iterate.
* Removed `Solver::abort()`. Now `Solver::updateState()` determines if aborting is appropriate based on the solver being `stalled` twice in a row.
* Along with the above changes, `Solver::reset()` is now only responsible for resetting the algorithm's (derived `Solver`'s) internal data.
* Updated unit tests to work with new changes.
* Updated readme.

## v1.1.0

* Changed default line search to LewisOvertion<Wolf::weak> so by default the `Lbfgsb` solver works on non-smooth functions.
* Added non-smooth test functions to unit tests.
* Updated readme.

## v1.0.0

* Initial release.

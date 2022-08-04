---
title: 'GlobalSensitivity.jl: Performant and Parallel Global Sensitivity Analysis with Julia'
tags:
  - julia
  - global sensitivity analysis
authors:
  - name: Vaibhav Kumar Dixit
    orcid: 0000-0001-7763-2717
    corresponding: true
    affiliation: 1
  - name: Christopher Rackauckas
    affiliation: "1, 2, 3"
affiliations:
 - name: Julia Computing
   index: 1
 - name: Massachusetts Institute of Technology
   index: 2
 - name: Pumas-AI
   index: 3
date: 21 June 2022
bibliography: paper.bib

---

# Summary

Global Sensitivity Analysis (GSA) methods are used to quantify the uncertainty in
output of a model with respect to the parameters. These methods allow practitioners to
measure both parameter's individual contributions and the contribution of their interactions
to the output uncertainity. GlobalSensitivity.jl is a Julia [@Bezanson2017] package containing implementation of some the most popular GSA methods. Currently it supports Delta Moment-Independent [@Borgonovo2007;@Plischke2013], DGSM [@Sobol2009], EASI [@Plischke2010;@Plischke2012], eFAST [@Saltelli1999;@Saltelli1998], Morris [@Morris1991;@Campolongo2007], Fractional Factorial [@Saltelli2008b], RBD-FAST [@Tarantola2006], Sobol [@Saltelli2008b;@Sobol2001;@Saltelli2002a] and regression based sensitivity [@Guido2016] methods.

# Statement of need

Global Sensitivity Analysis has become an essential part of modeling workflows for practitioners in various fields such as Quantitative Systems Pharmacology and Environmental Modeling [@saltelli2020five;@JAKEMAN2006602;@sher2022quantitative;@zhang2015sobol]. It can be used primarily in two stages, either before parameter estimation to simplify the fitting problem by fixing unimportant parameters or for analysis of the input parameters' influence on the output.

There are already some popular packages in R and Python, such as [sensitivity](https://cran.r-project.org/web/packages/sensitivity/index.html) and SALib [@Herman2017] for global sensitivity analysis. GlobalSensitivity.jl provides implementation of some of the popular GSA methods mentioned earlier, in Julia. Thus it benefits from the performance advantage of Julia, provides a convenient unified API for different GSA methods by leveraging multiple dispatch and has a parallelized implementation for some of the methods.

This package allows users to conveniently perform GSA on arbitrary functions and get the sensitivity analysis results and at the same time provides out of the box support for differential equations based models defined using the SciML interface [@Rackauckas2017DifferentialEquationsjlA;@RackauckasUDE].

## Examples

The following tutorials in the documentation [1](https://gsa.sciml.ai/stable/tutorials/juliacon21/) and [2](https://gsa.sciml.ai/stable/tutorials/parallelized_gsa/) cover workflows of using GlobalSensitivity.jl on the Lotka-Volterra differential equation, popularly known as the predator-prey model. We showcase how to use multiple GSA methods, analyse their results and leverage Julia's parallelism capabilities to perform global sensitivity analysis at scale. The plots have been created using the Makie.jl package [@DanischKrumbiegel2021], while many of the plots in the documentation use the Plots.jl package [@ChristPlots2022].

# Acknowledgements

This material is based upon work supported by the National Science Foundation under grant no.  OAC-1835443, grant no. SII-2029670,
grant no. ECCS-2029670, grant no. OAC-2103804, and grant no. PHY-2021825.  We also gratefully acknowledge the U.S. Agency for
International Development through Penn State for grant no. S002283-USAID. The information, data, or work presented herein was
funded in part by the Advanced Research Projects Agency-Energy (ARPA-E), U.S. Department of Energy, under Award Number DE-AR0001211
and DE-AR0001222. We also gratefully acknowledge the U.S. Agency for International Development through Penn State for grant no.
S002283-USAID. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States
Government or any agency thereof. This material was supported by The Research Council of Norway and Equinor ASA through Research
Council project "308817 - Digital wells for optimal production and drainage". Research was sponsored by the United States Air Force
Research Laboratory and the United States Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative
Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be
interpreted as representing the official policies, either expressed or implied, of the United States Air Force or the U.S. Government.
The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

# References

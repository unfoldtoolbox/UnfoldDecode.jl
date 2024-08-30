## Summary of our work

#### Getting familiar with EEG data and Unfold 
First, we familiarized ourselves with a simple example and thus enhanced our basic understanding of EEG data and plotting and Unfold. (src/EEG.jl)

#### Reading related papers
Secondly, we read and fully understood the concepts and ideas of BacktoBack regression from related papers, which helped us through the coding and documentationing for the b2b solver. 

#### Building vivid example for easy understanding 
Thirdly, we built an vivid example of a model of dogs, cats, vegetables etc. to help the users understand the basic ideas of what BacktoBack does, that is, to eliminate the effect of correlated (confounding) variable. And we successfully applied the BacktoBack regression on it. (src/backtoback.jl, docs/literate/BackToBack/About_BacktoBack.jl)

#### Adding serval regression functions to b2b_solver 
Forthly, we added several new regression functions(LS, Lasso, SVM, Adaboost) and revised the ridge function in the b2b solver algorithms. (src/b2b.jl)

#### Comparing the results 
Lastly, we compared the results of different functions. And SVM would be the most ideal one. (docs/literate/BackToBack/About_BacktoBack.jl)

#### Building a documentation
- We provide a quick-start tutorial for people who not familiar with b2b solver to quickly get start. (docs/literate/BackToBack/Quick_Start_b2b.jl)
- We provide a detailed explanation of the function of BacktoBack regression with a vivid example. (docs/literate/BackToBack/About_BacktoBack.jl)
- We provide detailed comments for b2b_solver. (docs/literate/BackToBack/About_b2b.jl)
- However, we encountered a problem that the documentation can only be seen in Live Server, but did not shown on the github. We tried to modified make.jl, but it did not work.

#### Further working
- We tried to do a unittest (test/runtests.jl), in which we built a simplified model, and compared the results using functions in backtoback regression in Julia to ones using sklearn in python (test/sklearn.py). But we encountered some problem with fitting. We tried by adding the test_utilities.jl and new_testcases.m from tests in Unfold.
- Also, we tried the LIBSVM in MLJ (src/0801B2BLIBSVM.jl), but it didn't work quite well, so we used LinearSVM instead. 
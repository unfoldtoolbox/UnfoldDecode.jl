using UnfoldDecode
using Test
using MLJLinearModels
using MLJ
using MLBase
using Tables
# include("test_utilities.jl")

using UnfoldSim

dat, evts = UnfoldSim.predef_eeg(; noiselevel = 0.1, return_epoched = true)
y_not = dat[1,1:10]
X = evts[1:10,:]

dat_3d = permutedims(repeat(y_not, 1, 1, 5), [3 1 2]); 
dat_3d .+= 0.1*rand(size(dat_3d)...);
# b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; cross_val_reps = 5);
@load SVMLinearRegressor pkg=MLJScikitLearnInterface
    model = MLJScikitLearnInterface.SVMLinearRegressor()
    G = Array{Float64}(undef,size(dat_3d,2),size(X,2))
    for pred in 1:size(X,2)
        mtm = machine(model,table(dat_3d),X[:,pred])
        MLBase.fit!(mtm,verbosity=0)
        G[:,pred] = fitted_params(mtm).coef
    end
    print(G) 

# data, evts = loadtestdata("test_case_2a") #
# println("sdf")
# @testset "UnfoldDecode.jl" begin
#     # Write your tests here.
    
#     # @testset "Iris Dataset" begin
#     #     # Load the iris dataset
#     #     # iris = dataset("datasets", "iris")
#     #     # X_iris = iris[:, 1:4]
#     #     # Y_iris = iris[:, 5]
#     #     # #print(X_iris)
#     #     # #print(Y_iris)
#     #     # println(size(Y_iris))
#     #     # println(Y_iris)
#     #     # println(Y_iris,1)
#     #     # println(size(Y_iris,1))
#     #     # println(Y_iris[1,:])
#     #     X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
#     #     y = [[-2, 1], [-1, 1], [-1, 2], [1, 10], [1, 20], [2, 10]]
#     #     X_test = [[-2,-1,-1,1,1,2],[-1,-1,-2,1,2,1]]
#     #     print(X_test[1])
#     #     println(size(X[:,1]))
#     #     # data = adjoint([-2 1;-1 1;-1 2;1 10;1 20;2 10])
#     #     data = adjoint(hcat(y))
#     #     println(data)
#     #     println(typeof(data))
#     #     println(table(data))
#     #     data=float(data)
#     #     println(typeof(data))
       


    

#     #     @test begin
#     #         @load SVMLinearRegressor pkg=MLJScikitLearnInterface
#     #         model = MLJScikitLearnInterface.SVMLinearRegressor()
#     #         G = Array{Float64}(undef,2,2)
#     #         for pred in 1:2
#     #             println(X[:,pred])
#     #             print("dfssssssssssssssssssssssssssssssssssssssssssssss")
#     #             mtm = machine(model,table(data),X[:,pred])
#     #             println("ddddddddddddddddddddddddddddddddddddddddddddddddd")
#     #             MLBase.fit!(mtm,verbosity=0)
#     #             println("sldkfjsiajflasidjdflaijsdkfj")
#     #             G[:,pred] = fitted_params(mtm).coef
#     #         end
#     #         print(G) 
#     #         @test result == [-0.11346491 -0.03184254;  0.25936799  0.53764103]
#     #     end
#     # end

#     # @testset "b2b_solver" begin
#     #     # Write your tests here.
#     #     @test begin
#     #         @load SVMLinearRegressor pkg=MLJScikitLearnInterface
#     #         model = MLJScikitLearnInterface.SVMLinearRegressor()
#     #         G = Array{Float64}(undef,2,2)
#     #         for pred in 1:2
#     #             println(X[:,pred])
#     # end

#     @testset "eeg" begin
#         dat, evts = UnfoldSim.predef_eeg(; noiselevel = 0.1, return_epoched = true);
# end

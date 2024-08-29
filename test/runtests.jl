using UnfoldDecode
using Test
using Pkg
Pkg.add("RDatasets")
using RDatasets
using MLJLinearModels
using MLJ
using MLBase
using Tables


@testset "UnfoldDecode.jl" begin
    # Write your tests here.
    @testset "Iris Dataset" begin
        # Load the iris dataset
        # iris = dataset("datasets", "iris")
        # X_iris = iris[:, 1:4]
        # Y_iris = iris[:, 5]
        # #print(X_iris)
        # #print(Y_iris)
        # println(size(Y_iris))
        # println(Y_iris)
        # println(Y_iris,1)
        # println(size(Y_iris,1))
        # println(Y_iris[1,:])
        X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
        y = [[-2, 1], [-1, 1], [-1, 2], [1, 10], [1, 20], [2, 10]]
        X_test = [[-2,-1,-1,1,1,2],[-1,-1,-2,1,2,1]]
        print(X_test[1])
        println(size(X[:,1]))
        # data = adjoint([-2 1;-1 1;-1 2;1 10;1 20;2 10])
        data = adjoint(hcat(y))
        println(data)
        println(typeof(data))
        println(table(data))
        data=float(data)
        println(typeof(data))
       


    

        @test begin
            @load SVMLinearRegressor pkg=MLJScikitLearnInterface
            model = MLJScikitLearnInterface.SVMLinearRegressor()
            G = Array{Float64}(undef,2,2)
            for pred in 1:2
                println(X[:,pred])
                print("dfssssssssssssssssssssssssssssssssssssssssssssss")
                mtm = machine(model,table(data),X[:,pred])
                println("ddddddddddddddddddddddddddddddddddddddddddddddddd")
                MLBase.fit!(mtm,verbosity=0)
                println("sldkfjsiajflasidjdflaijsdkfj")
                G[:,pred] = fitted_params(mtm).coef
            end
            print(G) 
            @test result == [-0.11346491 -0.03184254;  0.25936799  0.53764103]
        end
    end
end

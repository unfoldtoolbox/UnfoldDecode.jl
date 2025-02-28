
# setup some data to test

dat, evts = UnfoldSim.predef_eeg(; noiselevel = 0.1, return_epoched = true)

dat_3d = permutedims(repeat(dat, 1, 1, 3), [3 1 2]);
dat_3d .+= range(5, 20, size(dat_3d, 1)) .* rand(size(dat_3d)...); # vary the noise per channel

f = @formula(0 ~ 1 + condition + continuous)
designDict = [Any => (f, range(0, 0.44, step = 1 / 100))]

@testset "b2b tests" begin
    b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; cross_val_reps = 5)
    m = Unfold.fit(UnfoldModel, designDict, evts, dat_3d; solver = b2b_solver)
    @test size(coef(m)) == (1, 45, 3)


end


@testset "b2b tests" begin
    b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; cross_val_reps = 5)
    m = Unfold.fit(UnfoldModel, designDict, evts, dat_3d; solver = b2b_solver)
    @test size(coef(m)) == (1, 45, 3)
end

@testset "b2b algorithms" begin
    using UnfoldDecode: model_ridge, model_lsq, model_xgboost
    for g in [model_ridge, model_lsq, model_xgboost]
        for h in [model_ridge, model_lsq]
            b2b_solver =
                (x, y) -> UnfoldDecode.solver_b2b(
                    x,
                    y;
                    solver_G = g,
                    solver_H = h,
                    cross_val_reps = 1,
                )
            m = Unfold.fit(UnfoldModel, designDict, evts, dat_3d; solver = b2b_solver)
            @test isapprox(coef(m)[1, 15, 2], 0.5, atol = 0.1)
            @test isapprox(coef(m)[1, 5, 2], 0.0, atol = 0.1)
        end
    end


end

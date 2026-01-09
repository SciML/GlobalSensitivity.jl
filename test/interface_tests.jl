using Test
using GlobalSensitivity
using QuasiMonteCarlo
using Distributions
using LinearAlgebra

# Test functions
function ishi(X)
    A = 7
    B = 0.1
    sin(X[1]) + A * sin(X[2])^2 + B * X[3]^4 * sin(X[1])
end

function f_linear(X)
    X[1] + 2 * X[2] + 3 * X[3]
end

@testset "Interface Compatibility Tests" begin
    @testset "Float32 Support" begin
        @testset "Sobol with Float32 matrices" begin
            samples = 100
            lb = Float32[-π, -π, -π]
            ub = Float32[π, π, π]
            A = QuasiMonteCarlo.sample(samples, lb, ub, QuasiMonteCarlo.SobolSample())
            B = QuasiMonteCarlo.sample(samples, lb, ub, QuasiMonteCarlo.SobolSample())

            res = gsa(ishi, Sobol(), A, B)
            @test res.S1 isa Vector
            @test length(res.S1) == 3
        end

        @testset "RegressionGSA X,Y with Float32" begin
            samples = 100
            X = Float32.(
                QuasiMonteCarlo.sample(
                    samples,
                    [-1.0, -1.0, -1.0],
                    [1.0, 1.0, 1.0],
                    QuasiMonteCarlo.SobolSample()
                )
            )
            Y = Float32.(reshape([f_linear(X[:, i]) for i in 1:samples], 1, samples))

            res = gsa(X, Y, RegressionGSA())
            @test eltype(res.pearson) == Float32
        end
    end

    @testset "BigFloat Support" begin
        @testset "Sobol with BigFloat matrices" begin
            samples = 50
            lb = BigFloat[-π, -π, -π]
            ub = BigFloat[π, π, π]
            A = QuasiMonteCarlo.sample(samples, lb, ub, QuasiMonteCarlo.SobolSample())
            B = QuasiMonteCarlo.sample(samples, lb, ub, QuasiMonteCarlo.SobolSample())

            res = gsa(ishi, Sobol(), A, B)
            @test eltype(res.S1) == BigFloat
            @test length(res.S1) == 3
        end
    end

    @testset "Standard Float64 Methods" begin
        p_range = [[-π, π], [-π, π], [-π, π]]

        @testset "Morris method" begin
            res = gsa(
                ishi,
                Morris(num_trajectory = 5, len_design_mat = 4),
                p_range
            )
            @test res.means isa Matrix
            @test size(res.means, 2) == 3
        end

        @testset "eFAST method" begin
            res = gsa(ishi, eFAST(), p_range, samples = 300)
            @test res.S1 isa AbstractArray
            @test length(res.S1) == 3
        end

        @testset "DeltaMoment method" begin
            res = gsa(ishi, DeltaMoment(nboot = 10), p_range; samples = 200)
            @test length(res.deltas) == 3
        end

        @testset "EASI method" begin
            res = gsa(ishi, EASI(), p_range; samples = 300)
            @test res.S1 isa Vector
            @test length(res.S1) == 3
        end

        @testset "RSA method" begin
            res = gsa(ishi, RSA(), p_range; samples = 300)
            @test res.S isa AbstractVector
            @test length(res.S) == 3
        end

        @testset "MutualInformation method" begin
            res = gsa(ishi, MutualInformation(n_bootstraps = 50), p_range; samples = 300)
            @test res.S isa Vector
            @test length(res.S) == 3
        end

        @testset "RBDFAST method" begin
            res = gsa(ishi, RBDFAST(), num_params = 3, samples = 300)
            @test res isa Vector
            @test length(res) == 3
        end

        @testset "FractionalFactorial method" begin
            res = gsa(f_linear, FractionalFactorial(), num_params = 3)
            @test res[1] isa Vector  # main_effects
            @test length(res[1]) >= 3
        end
    end

    @testset "DGSM method" begin
        dist = [Uniform(4, 10), Normal(4, 1), Beta(2, 3)]
        res = gsa(f_linear, DGSM(), dist, samples = 200)
        @test res.a isa Vector
        @test length(res.a) == 3
    end

    @testset "Direct X,Y interface" begin
        samples = 200
        lb = Float64[-π, -π, -π]
        ub = Float64[π, π, π]
        X = QuasiMonteCarlo.sample(samples, lb, ub, QuasiMonteCarlo.SobolSample())
        Y = [ishi(view(X, :, i)) for i in 1:samples]

        @testset "DeltaMoment X,Y" begin
            res = gsa(X, Y, DeltaMoment(nboot = 10))
            @test length(res.deltas) == 3
        end

        @testset "EASI X,Y" begin
            res = gsa(X, Y, EASI())
            @test length(res.S1) == 3
        end

        @testset "RegressionGSA X,Y" begin
            Y_mat = reshape([f_linear(X[:, i]) for i in 1:samples], 1, samples)
            res = gsa(X, Y_mat, RegressionGSA())
            @test size(res.pearson) == (1, 3)
        end
    end
end

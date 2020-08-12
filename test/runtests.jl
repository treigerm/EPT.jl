using ExpectationProgramming
using Turing
using Test
using Random

#rng = MersenneTwister(42)
Random.seed!(42)

@testset "ExpectationProgramming.jl" begin
    @testset "Expectation Macro" begin
        @expectation function expct(y)
            x ~ Normal(0, 1) 
            y ~ Normal(x, 1)
            return x^2
        end

        yval = 1
        expct_conditioned = expct(yval)

        xval = 2
        vi = Turing.VarInfo(expct_conditioned.gamma1_pos)
        vi[@varname(x)] = [xval;]
        #vi[@varname(y)] = [yval;]

        # Check that the three different models return the right score for a given trace.
        gamma2_lp = logpdf(Normal(0, 1), xval) + logpdf(Normal(xval, 1), yval) 
        @test expct_conditioned.gamma2(vi) == xval^2
        @test Turing.getlogp(vi) == gamma2_lp

        gamma1_pos_lp = gamma2_lp + log(max(xval^2, 0))
        @test expct_conditioned.gamma1_pos(vi) == xval^2
        @test Turing.getlogp(vi) == gamma1_pos_lp

        gamma1_neg_lp = gamma2_lp + log(-min(xval^2, 0))
        @test expct_conditioned.gamma1_neg(vi) == xval^2 # TODO: Should it return 0 instead?
        @test Turing.getlogp(vi) == gamma1_neg_lp
    end

    @testset "Broken test" begin
        """
        This does not work because somehow I haven't got it to work that the 
        code in the function body gets evaluated in the scope of the user module.
        So here when calling expct() f is not in scope, even though it should 
        work.
        """

        #f(x) = x^2
        #@expectation function expct()
        #    x ~ Normal(0, 1) 
        #    y ~ Normal(x, 1)
        #    return f(x)
        #end
        
        #xval = 2
        #yval = 1
        #vi = Turing.VarInfo(expct.gamma1_pos())
        #vi[@varname(x)] = [xval;]
        #vi[@varname(y)] = [yval;]
    end

    @testset "Expectation Estimation" begin
        @expectation function expct(y)
            x ~ Normal(0, 1) 
            y ~ Normal(x, 1)
            return x^2
        end

        yval = 3
        expct_conditioned = expct(yval)

        num_annealing_dists = 10
        num_samples = 10

        tabi = TABI(
            AIS(num_annealing_dists, num_samples, SimpleRejection())
        )

        expct_estimate, diagnostics = estimate(expct_conditioned, tabi)
        @test !isnan(expct_estimate)
    end

    # Comment this out because it takes a while.
    #@testset "Convergence test" begin
    #    @expectation function expct(y)
    #        x ~ Normal(0, 1) 
    #        y ~ Normal(x, 1)
    #        return x
    #    end
    #    
    #    yval = 3
    #    expct_conditioned = expct(yval)

    #    num_annealing_dists = 100
    #    num_samples = 1000

    #    tabi = TABI(
    #        AIS(num_samples, num_annealing_dists)
    #    )

    #    expct_estimate, diagnostics = estimate(expct_conditioned, tabi)
    #    @test_broken isapprox(expct_estimate, 1.5, atol=1e-2)
    #end

    @testset "Diagnostics" begin
        @expectation function expct(y)
            x ~ Normal(0, 1) 
            y ~ Normal(x, 1)
            return x
        end

        yval = 3
        expct_conditioned = expct(yval)

        num_annealing_dists = 10
        num_samples = 10

        tabi = TABI(
            AIS(num_samples, num_annealing_dists, SimpleRejection())
        )

        expct_estimate, diagnostics = estimate(
            expct_conditioned, 
            tabi;
            store_intermediate_samples=true
        )

        keys = [:Z2_info, :Z1_negative_info, :Z1_positive_info]

        for k in keys
            @test haskey(diagnostics, k)

            @test typeof(diagnostics[k][:ess]) == Float64
            @test typeof(diagnostics[k][:Z_estimate]) == Float64
            @test size(diagnostics[k][:samples]) == (num_samples,)
            @test haskey(diagnostics[k], :intermediate_samples)
        end
    end

    @testset "Rejection Samplers" begin
        @expectation function expct(y)
            x ~ Normal(0, 1) 
            y ~ Normal(x, 1)
            return x
        end

        yval = 3
        expct_conditioned = expct(yval)

        num_annealing_dists = 10
        num_samples = 10

        tabi_no_rejection = TABI(
            AIS(num_samples, num_annealing_dists, SimpleRejection())
        )
        _, _ =  estimate(
            expct_conditioned, 
            tabi_no_rejection;
            store_intermediate_samples=true
        )

        tabi_rejection = TABI(
            AIS(num_samples, num_annealing_dists, RejectionResample())
        )
        _, _ =  estimate(
            expct_conditioned, 
            tabi_rejection;
            store_intermediate_samples=true
        )
    end

    @testset "Disable Z1_pos or Z1_neg" begin
        @expectation function expct(y)
            x ~ Normal(0, 1) 
            y ~ Normal(x, 1)
            return x^2
        end

        yval = 3
        expct_conditioned = expct(yval)

        num_annealing_dists = 10
        num_samples = 2

        tabi_no_Z1_neg = TABI(
            AIS(num_samples, num_annealing_dists, SimpleRejection()),
            AIS(0, num_annealing_dists, SimpleRejection()),
            AIS(num_samples, num_annealing_dists, SimpleRejection())
        )

        _, d =  estimate(
            expct_conditioned, 
            tabi_no_Z1_neg;
            store_intermediate_samples=true
        )

        full_tabi = TABI(AIS(num_samples, num_annealing_dists, SimpleRejection()))
        _, d_full =  estimate(
            expct_conditioned, 
            tabi_no_Z1_neg;
            store_intermediate_samples=true
        )

        # Check that estimate is type-stable.
        @test typeof(d_full) == typeof(d)
    end
end

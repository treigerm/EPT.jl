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
            AIS(num_annealing_dists, num_samples)
        )

        expct_estimate = estimate(expct_conditioned, tabi)
        @test !isnan(expct_estimate)
    end

    # TODO: TABI convergence test.
end

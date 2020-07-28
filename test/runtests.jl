using ExpectationProgramming
using Turing
using Test
using Random

#rng = MersenneTwister(42)
Random.seed!(42)

@testset "ExpectationProgramming.jl" begin
    @expectation function expct()
        x ~ Normal(0, 1) 
        y ~ Normal(x, 1)
        return x^2
    end

    xval = 2
    yval = 1
    vi = Turing.VarInfo(expct.gamma1_pos())
    vi[@varname(x)] = [xval;]
    vi[@varname(y)] = [yval;]

    gamma2_lp = logpdf(Normal(0, 1), xval) + logpdf(Normal(xval, 1), yval) 
    @test expct.gamma2()(vi) == xval^2
    @test Turing.getlogp(vi) == gamma2_lp

    gamma1_pos_lp = gamma2_lp + log(max(xval^2, 0))
    @test expct.gamma1_pos()(vi) == xval^2
    @test Turing.getlogp(vi) == gamma1_pos_lp

    gamma1_neg_lp = gamma2_lp + log(-min(xval^2, 0))
    @test expct.gamma1_neg()(vi) == xval^2 # TODO: Should it return 0 instead?
    @test Turing.getlogp(vi) == gamma1_neg_lp
end
